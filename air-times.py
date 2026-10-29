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
sched = C.load("air-schedule-oag")
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
    C.save(sched, "air-schedule-oag")
sched = sched.drop(columns=["nstops", "seats"], errors="ignore")#.view()

#%% Airport names, codes, locations & time zone offsets
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
)
airports["name"] = airports["name"].str.replace(" Airport", "")
airports["node_id"] = "AIR_" + airports["iata"]
ref_date = datetime.combine(C.PARAMS["AIR_TIMEZONE_DATE"], time(12))
airports["utc_offset"] = airports["icc"].map(
    C.load("countries").set_index("icc")["timezone"]
    .apply(lambda tz: np.int16(ref_date.replace(tzinfo=ZoneInfo(tz))
                               .utcoffset().total_seconds() / 60))
)

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

C.save(airports2, "airports")

#%% Timetable links [4s]
tt = sched[(sched["src"].isin(airports2["iata"])) &
           (sched["trg"].isin(airports2["iata"]))]
tz_offsets = airports2.set_index("iata")["utc_offset"]
tt["dep"] = np.int16(tt["dep"] - tt["src"].map(tz_offsets))
tt["arr"] = np.int16(tt["arr"] - tt["trg"].map(tz_offsets))
keys = ["start_date", "end_date", "op_days"]
days = tt[keys].drop_duplicates(ignore_index=True)
start_date = pd.Timestamp(C.PARAMS["SERVICE_START"])
end_date = pd.Timestamp(C.PARAMS["SERVICE_END"])
dates = pd.date_range(start_date, end_date)
days = days[days["end_date"].astype(str) >= str(start_date)]
start_values = days["start_date"].to_numpy(dtype="datetime64[D]")
end_values = days["end_date"].to_numpy(dtype="datetime64[D]")
date_values = dates.to_numpy(dtype="datetime64[D]")
active = np.column_stack([
    days["op_days"].str.contains(str(d)).to_numpy(dtype=bool)
    for d in range(1, 8)
])[:, dates.dayofweek]
active &= (date_values[None, :] >= start_values[:, None])
active &= (date_values[None, :] <= end_values[:, None])
date_ids = np.arange(len(dates), dtype=np.int32)
days["date"] = [date_ids[row] for row in active]
days = days[days["date"].str.len() > 0]
tt = tt.merge(days, on=keys).drop(columns=keys)
tt = tt[tt["src"].astype(str) != tt["trg"].astype(str)]
tt = tt.explode("date", ignore_index=True).astype({"date": np.int16})
tt["src"] = tt["src"].map(lambda x: "AIR_" + x)
tt["trg"] = tt["trg"].map(lambda x: "AIR_" + x)
tt.view();

#%% Major airlines/carriers
tt["operator"] = tt["carrier"].cat.rename_categories(dict(
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
)).cat.remove_unused_categories()

C.save(tt, "air-timetable")

#%% Graph links (with distance, travel time & frequency)
links = (
    tt.assign(time=tt["arr"] - tt["dep"])
    .groupby(["src", "trg", "operator"], observed=True)
    ["time"].agg(["count", "mean", "median", "std"])
    .set_axis(["nservices", "avg_time", "median_time", "std_time"], axis=1)
    .astype(np.float32).astype({"nservices": np.int32})
    .reset_index()
)
total_days = (C.PARAMS["SERVICE_END"] - C.PARAMS["SERVICE_START"]).days + 1
links["avg_freq"] = np.float32(links["nservices"] / total_days)

# Great Circle Distance (km)
ap_xy = airports2.set_index("node_id").get_coordinates().reset_index()
links = links.merge(ap_xy.set_axis(["src", "x0", "y0"], axis=1))
links = links.merge(ap_xy.set_axis(["trg", "x1", "y1"], axis=1))
links["dist"] = np.int16(haversine.haversine_vector(
    list(zip(links.pop("y0"), links.pop("x0"))),
    list(zip(links.pop("y1"), links.pop("x1"))),
))
links.view();

C.save(links, "air-links")
