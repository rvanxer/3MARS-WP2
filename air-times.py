"""Use the OAG flight schedule data for air network construction."""

#%% Imports
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

params = C.load_params()

#%% OAG flight schedules [14s]
sched = C.load("air-timetable")
if sched is None:
    cols = (
        ("DEP_AIRPORT_CODE", "src", "category"),
        ("ARR_AIRPORT_CODE", "trg", "category"),
        ("AIRLINE_CODE", "carrier", "category"),
        ("FLIGHT_NO", "flight", np.int16),
        ("DEPARTURE_TIME", "dep", np.int16),
        ("ARRIVAL_TIME", "arr", np.int16),
        ("STOPS", "nstops", np.int8),
        ("LOCALDAYSOFOP", "op_days", "category"),
        ("SEATS_ECO", "seats", np.int16),
        ("EFFFROM", "start_date", "category"),
        ("EFFTO", "end_date", "category"),
    )
    sched = (
        pd.read_csv(C.DATA / "oag-schedules.zip",
                    usecols=[c[0] for c in cols])
        .rename(columns={c[0]: c[1] for c in cols})
        .astype({c[1]: c[2] for c in cols})
    )
    # add 24 h for next-day schedule
    sched.loc[sched["dep"] >= sched["arr"], "arr"] += 2400
    # convert times to minutes since start of day
    for col in ["arr", "dep"]:
        sched[col] = np.int16((sched[col] // 100) * 60 + (sched[col] % 100))
    sched = sched.drop_duplicates(ignore_index=True)
    C.save(sched, "air-timetable")
sched.view();

#%% Airport codes & locations
airports = (
    pd.read_csv("https://raw.githubusercontent.com/ip2location/"
                "ip2location-iata-icao/master/iata-icao.csv")
    .rename(columns={"airport": "name"})
    .pipe(C.pdf2gdf, "longitude", "latitude", C.CRS_DEG)
    .sjoin(C.load("countries")[["icc", "geometry"]])
    [["iata", "icao", "name", "icc", "geometry"]]
    .drop_duplicates("iata", ignore_index=True)
    .merge(sched[["src", "trg"]]
           .melt(value_name="iata")
           ["iata"].drop_duplicates(), on="iata")
).view()

#%% Map to urban areas
fua_centroids = (
    C.load("fuas")
    [["id", "centre"]]
    .set_axis(["fua", "geometry"], axis=1)
    .pipe(gpd.GeoDataFrame, crs=C.CRS_DEG)
    .to_crs(C.CRS_EU)
).view()

airports2 = airports.merge(
    airports.set_index("iata")
    .to_crs(C.CRS_EU)
    .buffer(params.AIRPORT_CATCH_RADIUS * 1000)
    .rename("geometry").reset_index()
    .sjoin(fua_centroids)
    .sort_values("fua")
    .groupby("iata")
    ["fua"].agg(list),
    on="iata"
).view()
airports2["name"] = airports2["name"].str.replace(" Airport", "")

C.save(airports2, "airports")

#%% Timetable with flight count per flight [8s]
tt = (
    sched[(sched["src"].isin(airports2["iata"])) &
          (sched["trg"].isin(airports2["iata"]))]
    .astype({"start_date": str, "end_date": str})
    .reset_index(drop=True).rename_axis("row")
)
days = (
    tt.reset_index()
    .groupby(["start_date", "end_date", "op_days"])
    ["row"].agg(list).reset_index()
)
days["op_days"] = [
    [int(x) - 1 for x in r.replace(" ", "")] for r in days["op_days"]
]
days["nflights"] = days.apply(lambda r:
    pd.date_range(r["start_date"], r["end_date"])
    .day_of_week.isin(r["op_days"]).sum(), axis=1
)
days = days[["row", "nflights"]].explode("row").astype({"row": int})
tt = tt.merge(days, on="row").set_index("row").view(1)

#%% Travel time & frequency by carrier
od_tt = (
    tt.assign(time=(tt["arr"] - tt["dep"]) * tt["nflights"])
    .groupby(["carrier", "src", "trg"], observed=True)
    .agg({"time": "sum", "nflights": "sum",
          "start_date": "min", "end_date": "max"})
    .reset_index()
)
od_tt["ndays"] = [len(pd.date_range(*x)) for x in zip(
    od_tt["start_date"], od_tt["end_date"])]
od_tt["time"] /= od_tt["nflights"]
od_tt["freq"] = od_tt["nflights"] / od_tt["ndays"]
od_tt = od_tt[["src", "trg", "carrier", "time", "freq",
               "nflights", "ndays"]].view()

C.save(od_tt, "air-links")

#%%
