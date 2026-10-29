"""Create intracity interhub PT links using global GTFS data. [32s]"""

#%% Imports
from datetime import datetime, time
from zoneinfo import ZoneInfo

import geopandas as gpd
from numba import njit
import numpy as np
from numpy import int16 as I16
from numpy import int32 as I32
import pandas as pd

import config as C

#%% Nodes: Cities and transport hubs
fuas = C.load("fuas").set_index("id").rename_axis("fua")#.view()

# one row per airport-FUA or station-FUA pair
hubs = (
    pd.concat([
        (C.load("airports")
         [["node_id", "fua", "geometry"]]
         .assign(kind="Airport")),
        (C.load("ic-stations")
         [["node_id", "geometry"]]
         .to_crs(fuas.crs)
         .sjoin(fuas[["geometry"]])
         .assign(fua=lambda df: [[x] for x in df["fua"]])
         .assign(kind="Station")),
    ], ignore_index=True)
    [["node_id", "kind", "fua", "geometry"]]
    .rename_axis("hub")
)
hubs = pd.concat([hubs, hubs.get_coordinates()], axis=1)#.view()

#%% Stop-to-hub mapping [7s]
C.log("Identifying stops near hubs")
stops = (
    C.load("gtfs/db-stops", columns=["fid", "id", "lon", "lat"])
    .dropna(subset=["lon", "lat"]).reset_index(drop=True)
    .rename_axis("stop_id")
)#.view()

stop2hub = (
    gpd.sjoin_nearest(
        C.pdf2gdf(stops, crs=C.CRS_DEG)[["geometry"]].to_crs(C.CRS_EU),
        hubs["geometry"].reset_index().to_crs(C.CRS_EU),
        max_distance=C.PARAMS["STN_BUFFER_RADIUS"],
        distance_col="dist"
    ).reset_index()
    [["stop_id", "hub", "dist"]]
    .sort_values("dist").drop(columns="dist")
    .drop_duplicates("stop_id", ignore_index=True)
    .merge(stops[["fid", "id"]], on="stop_id")
    .astype({"stop_id": I32, "hub": I16})
)#.view()

#%% Trips with time zone adjustment [4s]
C.log("Loading trips (with time zone offsets)")
routes = (
    C.load("gtfs/db-routes")
    .rename(columns={"id": "route_id"})
    [["fid", "route_id", "tz"]]
    .dropna(subset="tz")
    .pipe(lambda df: df[df["tz"].str.split("/").str[0] == "Europe"])
    .astype({"tz": "category"})
)
ref_date = datetime.combine(C.PARAMS["AIR_TIMEZONE_DATE"], time(12))
routes["utc_offset"] = routes["tz"].map({
    tz: I16(ref_date.replace(tzinfo=ZoneInfo(tz))
                 .utcoffset().total_seconds() / 60)
    for tz in routes["tz"].cat.categories
})
trips = (
    C.load("gtfs/db-trips")
    .rename(columns={"id": "trip_id"})
    .merge(routes, on=["fid", "route_id"])
    .assign(start=lambda df: df["start"] - df["utc_offset"])
    [["fid", "trip_id", "start", "dateset_id", "stopseq_id", "timeseq_id"]]
    .rename_axis("trip").reset_index()
    .astype({"trip": I32, "trip_id": I32, "start": I32})
)#.view()

#%% Trips with hubs [12s]
C.log("Filtering trips covering hub stops")
stopseq = (
    C.load("gtfs/db-stop_seq")
    .rename(columns={"id": "stopseq_id"})
    .explode("stop_id").astype({"stop_id": I32})
    .assign(snum=lambda df: I16(df.groupby(
        ["fid", "stopseq_id"], sort=False).cumcount()))
    .merge(stop2hub[["fid", "id", "hub"]]
           .rename(columns={"id": "stop_id"}),
           on=["fid", "stop_id"])
    .drop(columns="stop_id")
    .loc[lambda df: (df.groupby(["fid", "stopseq_id"], sort=False)
                     ["snum"].transform("size") > 1)]
    .groupby(["fid", "stopseq_id"], sort=False)
    [["snum", "hub"]].agg(list).reset_index()
)#.view()

trips = (
    trips.merge(stopseq[["fid", "stopseq_id"]]
                .drop_duplicates(), on=["fid", "stopseq_id"])
)#.view()

#%% Trips in service period [4s]
C.log("Filtering trips in service period")
dates = (
    C.load("gtfs/db-datesets")
    .rename(columns={"id": "dateset_id", "day_id": "date"})
    .merge(trips[["fid", "dateset_id"]].drop_duplicates(),
           on=["fid", "dateset_id"])
    .explode("date").astype({"date": I32})
)
dates["date"] += 20200101 # add reference date to map to "%Y%m%d"
dates = dates.query("{} <= date <= {}".format(*[
    int(C.PARAMS[x].strftime("%Y%m%d"))
    for x in ["SERVICE_START", "SERVICE_END"]
])).reset_index(drop=True)
cur_dates = dates["date"].unique()
new_dates = (pd.to_datetime(cur_dates.astype(str), format="%Y%m%d")
             - pd.Timestamp(C.PARAMS["SERVICE_START"])).days
dates["date"] = I16(dates["date"].map(dict(zip(cur_dates, new_dates))))
datesets = (
    dates.sort_values("date")
    .groupby(["fid", "dateset_id"])
    ["date"].agg(list).reset_index()
    .rename_axis("dateset")
)#.view()
del dates

trips = (
    trips.merge(datesets.reset_index(), on=["fid", "dateset_id"])
    .drop(columns=["dateset_id"]).astype({"dateset": I32})
)#.view()

#%% Timetable [11s]
C.log("Preparing inter-hub timetable")
tt = (
    trips[["fid", "timeseq_id", "stopseq_id"]].drop_duplicates()
    .merge(C.load("gtfs/db-time_seq",
                  columns=["fid", "id", "arr_time", "wait"])
           .rename(columns={"id": "timeseq_id", "arr_time": "arr"}),
           on=["fid", "timeseq_id"])
    .merge(stopseq, on=["fid", "stopseq_id"])
)
for col in ["arr", "wait"]:
    tt[col] = [times[idx] for times, idx in zip(tt[col], tt["snum"])]
tt = (
    tt.drop(columns="snum")
    .explode(["hub", "arr", "wait"], ignore_index=True)
    .astype({"hub": I16, "arr": I32, "wait": I32})
    .merge(trips, on=["fid", "timeseq_id", "stopseq_id"])
    .astype({"trip": I32}).drop(columns="trip_id")
)
# filter reasonable arrival/dep times in minutes since day start
tt["arr"] = I16((tt["arr"] + tt["start"]) / 60)
tt["dep"] = I16(tt["arr"] + tt["wait"] / 60)
tt = (
    tt.query("0 <= arr <= dep <= 24 * 60")
    .groupby(["trip", "fid", "dateset", "hub"], sort=False)
    [["arr", "dep"]].median()
    .astype(I16).reset_index()
    .sort_values(["trip", "arr"], ignore_index=True)
)#.view()

#%% P-space links [5s]
C.log("Creating P-space links")
@njit
def get_timetable_pairs(trip, stn, arr, dep):
    """Expand ordered journey stops into P-space timetable rows."""
    n_pairs = 0
    start = 0
    while start < len(trip):
        end = start + 1
        while end < len(trip) and trip[end] == trip[start]:
            end += 1
        n_stn = end - start
        n_pairs += n_stn * (n_stn - 1) // 2
        start = end
    _trip = np.empty(n_pairs, dtype=trip.dtype)
    _src = np.empty(n_pairs, dtype=stn.dtype)
    _trg = np.empty(n_pairs, dtype=stn.dtype)
    _dep = np.empty(n_pairs, dtype=dep.dtype)
    _arr = np.empty(n_pairs, dtype=arr.dtype)
    k = 0
    start = 0
    while start < len(trip):
        end = start + 1
        while end < len(trip) and trip[end] == trip[start]:
            end += 1
        for i in range(start, end - 1):
            for j in range(i + 1, end):
                _trip[k] = trip[start]
                _src[k], _trg[k] = stn[i], stn[j]
                _dep[k], _arr[k] = dep[i], arr[j]
                k += 1
        start = end
    return _trip, _src, _trg, _dep, _arr

tt2 = pd.DataFrame(dict(zip(
    ["trip", "src", "trg", "dep", "arr"],
    get_timetable_pairs(
        tt["trip"].values, tt["hub"].values,
        tt["arr"].values, tt["dep"].values
    )
))).query("src != trg and arr > dep").reset_index(drop=True)
# filter hub ODs within the same city (or mapped to the same city)
tt2 = (
    tt2[["src", "trg"]].drop_duplicates()
    .assign(src_fua=lambda df: df["src"].map(hubs["fua"]))
    .assign(trg_fua=lambda df: df["trg"].map(hubs["fua"]))
    .explode("src_fua")
    .explode("trg_fua")
    .query("src_fua == trg_fua")
    [["src", "trg"]].drop_duplicates()
    .merge(tt2, on=["src", "trg"])
)
tt2["dateset"] = tt2.pop("trip").map(trips.set_index("trip")["dateset"])
for col in ["src", "trg"]:
    tt2[col] = (pd.Categorical(tt2[col], categories=hubs.index)
                  .rename_categories(hubs["node_id"]))
tt2 = (
    tt2.merge(datesets["date"], on="dateset")
    .drop(columns="dateset")
    .explode("date").astype({"date": I16})
    .drop_duplicates(ignore_index=True)
).view()

C.save(tt2, "local-pt-timetable")

#%% Graph links
links = (
    tt2.assign(time=tt2["arr"] - tt2["dep"])
    .groupby(["src", "trg"], observed=True)
    ["time"].agg(["count", "mean", "median", "std"])
    .set_axis(["nservices", "avg_time", "median_time", "std_time"], axis=1)
    .astype(np.float32).astype({"nservices": np.int32})
    .reset_index()
)
total_days = (C.PARAMS["SERVICE_END"] - C.PARAMS["SERVICE_START"]).days + 1
links["avg_freq"] = np.float32(links["nservices"] / total_days)
links.view();

C.save(links, "local-pt-links")

#%%
