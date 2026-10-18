"""Use the OAG flight schedule data for air network construction."""

#%% Imports
from datetime import datetime, time
from zoneinfo import ZoneInfo

import haversine
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

#%% OAG flight schedules [14s]
sched = C.load("air-timetable")
if sched is None:
    cols = (
        ("DEP_AIRPORT_CODE", "src", "category"),
        ("ARR_AIRPORT_CODE", "trg", "category"),
        ("AIRLINE_CODE", "carrier", "category"),
        ("FLIGHT_NO", "flight", "int16"),
        ("DEPARTURE_TIME", "dep", "int16"),
        ("ARRIVAL_TIME", "arr", "int16"),
        ("STOPS", "nstops", "int8"),
        ("LOCALDAYSOFOP", "op_days", "category"),
        ("SEATS_ECO", "seats", "int16"),
        ("EFFFROM", "start_date", "category"),
        ("EFFTO", "end_date", "category"),
    )
    sched = (
        pd.read_csv(C.DATA / "oag-schedules.zip",
                    usecols=[c[0] for c in cols])
        .rename(columns={c[0]: c[1] for c in cols})
        .astype({c[1]: c[2] for c in cols})
    )
    for col in ["arr", "dep"]:
        sched[col] = np.int16((sched[col] // 100) * 60 + (sched[col] % 100))
    sched = sched.drop_duplicates(ignore_index=True)
    C.save(sched, "air-timetable")
sched#.view();

#%% Airport codes & locations
airports = (
    pd.read_csv(C.URLS["airport-locations"])
    .rename(columns={"airport": "name"})
    .pipe(C.pdf2gdf, "longitude", "latitude", C.CRS_DEG)
    .sjoin(C.load("countries")[["icc", "geometry"]])
    [["iata", "icao", "name", "icc", "geometry"]]
    .drop_duplicates("iata", ignore_index=True)
    .merge(sched[["src", "trg"]]
           .melt(value_name="iata")
           ["iata"].drop_duplicates(), on="iata")
)#.view()

#%% Map to urban areas
fua_centroids = (
    C.load("fuas")
    [["id", "centre"]]
    .set_axis(["fua", "geometry"], axis=1)
    .pipe(gpd.GeoDataFrame, crs=C.CRS_DEG)
    .to_crs(C.CRS_EU)
)#.view()

airports2 = airports.merge(
    airports.set_index("iata")
    .to_crs(C.CRS_EU)
    .buffer(C.PARAMS["AIRPORT_CATCH_RADIUS"] * 1000)
    .rename("geometry").reset_index()
    .sjoin(fua_centroids)
    .sort_values("fua")
    .groupby("iata")
    ["fua"].agg(list),
    on="iata"
)#.view()
airports2["name"] = airports2["name"].str.replace(" Airport", "")
C.save(airports2, "airports")

#%% Timetable with flight count per flight [8s]
tt = (
    sched[(sched["src"].isin(airports2["iata"])) &
          (sched["trg"].isin(airports2["iata"]))]
    .astype({"start_date": str, "end_date": str})
    .reset_index(drop=True).rename_axis("row")
)#.view()
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

#%% Travel times, adjusted for time zone differences
ref_date = datetime.combine(C.PARAMS["AIR_TIMEZONE_DATE"], time(12))
cntr_tz = C.load("countries").set_index("icc")["timezone"]
airport_tz = airports.set_index("iata")["icc"].map(cntr_tz)
offsets = airport_tz.map(lambda tz:
    ref_date.replace(tzinfo=ZoneInfo(tz)).utcoffset().total_seconds() / 60
).astype(np.int16)
tt["dep_utc"] = (tt["dep"] - tt["src"].map(offsets)).astype(np.int16)
tt["arr_utc"] = (tt["arr"] - tt["trg"].map(offsets)).astype(np.int16)
tt["duration"] = (tt["arr_utc"] - tt["dep_utc"]).astype(np.int16)

#%% Airlines/carriers
carriers = dict(
    A3 = "Aegean Airlines",
    AF = "Air France",
    AY = "Finnair",
    AZ = "ITA Airways",
    BA = "British Airways",
    BT = "airBaltic",
    DY = "Norwegian Air Shuttle",
    EW = "Eurowings",
    FI = "Icelandair",
    FR = "Ryanair",
    HV = "Transavia",
    IB = "Iberia",
    KL = "KLM Royal Dutch Airlines",
    KM = "Air Malta",
    LG = "Luxair",
    LH = "Lufthansa",
    LO = "LOT Polish Airlines",
    LS = "Jet2.com",
    LX = "Swiss International Air Lines",
    OS = "Austrian Airlines",
    OU = "Croatia Airlines",
    QS = "Smartwings",
    RO = "TAROM",
    SK = "SAS Scandinavian Airlines",
    SN = "Brussels Airlines",
    TO = "Transavia France",
    TP = "TAP Air Portugal",
    U2 = "EasyJet",
    V7 = "Volotea",
    W6 = "Wizz Air",
)

#%% Link attributes: Distance, travel time & frequency by carrier
links = (
    tt.assign(time=tt["duration"] * tt["nflights"])
    .groupby(["carrier", "src", "trg"], observed=True)
    .agg({"time": "sum", "nflights": "sum",
          "start_date": "min", "end_date": "max"})
    .reset_index()
)
links["carrier"] = links["carrier"].map(carriers).astype("category")
links["ndays"] = [len(pd.date_range(*x)) for x in zip(
    links["start_date"], links["end_date"])]
links["time"] /= links["nflights"]
links["freq"] = links["nflights"] / links["ndays"]
links = links[["src", "trg", "carrier", "time", "freq",
               "nflights", "ndays"]]

ap_xy = airports.set_index("iata").get_coordinates().reset_index()
links = links.merge(ap_xy.set_axis(["src", "x0", "y0"], axis=1))
links = links.merge(ap_xy.set_axis(["trg", "x1", "y1"], axis=1))
links["dist"] = np.int16(haversine.haversine_vector(
    list(zip(links.pop("y0"), links.pop("x0"))),
    list(zip(links.pop("y1"), links.pop("x1"))),
))
links = links.dropna(ignore_index=True).view()

C.save(links, "air-links")
