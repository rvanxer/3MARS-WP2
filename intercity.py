"""Prepare intercity schedule data from the GTFS database."""

#%% Imports
from collections import Counter, defaultdict
from itertools import combinations

import geopandas as gpd
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from zoneinfo import ZoneInfo

I16, UI16, I32 = np.int16, np.uint16, np.int32

import config as C

params = C.load_params()

#%% Urban stops [3s]
C.log("Filtering urban stops")

fuas = C.load("fuas").rename(columns={"id": "fua"})#.view()

stops = (
    C.load("gtfs/db-stops")
    .rename(columns={"id": "stop", "fid": "feed"})
    .pipe(C.pdf2gdf, crs=C.CRS_DEG)
    .sjoin(fuas[["fua", "geometry"]], predicate="within")
    .drop(columns="index_right", errors="ignore")
)#.view()

#%% Routes
C.log("Identifying routes and computing timezone offsets")
routes = (
    C.load("gtfs/db-routes")
    .assign(rail=lambda df: df["mode_id"].map(
        {x: 0 for x in params.BUS_ROUTE_TYPES} |
        {x: 1 for x in params.RAIL_ROUTE_TYPES}
    )).dropna(subset="rail", ignore_index=True)
    .astype({"agency": "category", "rail": bool})
    .rename(columns={"id": "route", "fid": "feed"})
    [["feed", "route", "agency", "rail", "tz"]]
)
routes["tz"] = routes["tz"].astype("string").str.strip().str.strip("'\"")
tz_offsets = {}
for tz in routes["tz"].dropna().unique():
    try:
        tz_offsets[tz] = (
            datetime(2025, 1, 15, 12)
            .replace(tzinfo=ZoneInfo(tz))
            .utcoffset()
            .total_seconds() / 3600
        )
    except Exception as e:
        print(tz, e)
routes["tz_gap"] = routes.pop("tz").map(tz_offsets)
routes = routes.dropna(subset="tz_gap").astype({"tz_gap": I32})#.view()

#%% Datesets [6s]
C.log("Extracting datesets")
dates = (
    C.load("gtfs/db-datesets")
    .rename(columns={"id": "dateset_id", "fid": "feed"})
    .explode("day_id").astype({"day_id": np.int32})
)
date2int = lambda date: np.int32(str(date).replace("-", ""))
dates["date"] = dates.pop("day_id") + date2int(params.BASE_START_DATE)
dates = dates[dates["date"] >= date2int(params.SERVICE_START)]
dates = dates[dates["date"] <= date2int(params.SERVICE_END)]
row_codes, dateset_ids = pd.factorize(dates.index, sort=True)
col_codes, date_ids = pd.factorize(dates["date"], sort=True)
date_mat = np.zeros((len(dateset_ids), len(date_ids)), dtype=bool)
date_mat[row_codes, col_codes] = True
datesets = pd.concat([
    dates[["feed", "dateset_id"]]
    .drop_duplicates(ignore_index=True),
    pd.DataFrame(date_mat, columns=date_ids)
], axis=1).rename_axis("dateset")#.view()
del dates, date_mat

#%% Journeys [4s]
C.log("Preparing journeys")
jrn = (
    C.load("gtfs/db-trips")
    .rename(columns={"id": "trip", "fid": "feed", "route_id": "route",
                     "stopseq_id": "stopseq", "timeseq_id": "timeseq"})
    .merge(routes, on=["feed", "route"])
    .merge(datesets[["feed", "dateset_id"]].reset_index(),
           on=["feed", "dateset_id"])
    .drop(columns=["dateset_id", "route", "end"])
    .astype({"trip": I32, "dateset": I32})
    .sort_values(["feed", "trip"], ignore_index=True)
    .rename_axis("jrn")
)#.view() # 3s
datesets = datesets.drop(columns=["feed", "dateset_id"], errors="ignore")#.view()

#%% Stop sequences [6s]
C.log("Filtering intercity stop sequences")
stopseq = (
    C.load("gtfs/db-stop_seq")
    .rename(columns={"id": "stopseq", "fid": "feed", "stop_id": "stop"})
    .merge(jrn[["feed", "stopseq"]].drop_duplicates(), on=["feed", "stopseq"])
)#.view()

stopseq_ic = (
    stopseq.explode("stop")
    .astype({"stop": int})
    .merge(stops[["feed", "stop", "fua"]], on=["feed", "stop"])
    .groupby(["feed", "stopseq"])
    ["fua"].nunique()
    .pipe(lambda x: x[x >= 2])
    .reset_index()[["feed", "stopseq"]]
    .merge(stopseq, on=["feed", "stopseq"])
)#.view()

#%% Stations [2s]
C.log("Clustering terminal stops into 'stations'")
# Stops that serve as endpoints of at least one intercity route
df = stopseq_ic.set_index(["feed", "stopseq"])
termini = (
    pd.concat([df["stop"].str[0], df["stop"].str[-1]])
    .reset_index()
    .merge(jrn[["feed", "stopseq"]].drop_duplicates(),
           on=["feed", "stopseq"])
    [["feed", "stop"]].drop_duplicates()
    .merge(stops[["feed", "stop", "geometry"]], on=["feed", "stop"])
    .pipe(gpd.GeoDataFrame, crs=C.CRS_DEG)
)
stns = (
    termini.to_crs(C.CRS_EU)
    .set_index(["feed", "stop"])
    .get_coordinates().reset_index()
)
stns["stn"] = DBSCAN(
    eps=params.STOP_CLUSTER_RADIUS,
    min_samples=1,
    algorithm="ball_tree"
).fit_predict(stns[["x", "y"]])
stns = (
    stns.merge(stops[["feed", "stop", "name"]], on=["feed", "stop"])
    .groupby("stn")
    .agg({"feed": list, "stop": list, "x": "mean", "y": "mean",
          "name": lambda x: " | ".join(pd.Series(x).value_counts().index)})
    .pipe(C.pdf2gdf, x="x", y="y", crs=C.CRS_EU)
    .to_crs(C.CRS_DEG)
    [["name", "feed", "stop", "geometry"]]
    .sjoin(fuas[["fua", "geometry"]])
    [["name", "fua", "feed", "stop", "geometry"]]
)#.view()

#%% Intercity journeys
C.log("Filtering intercity journeys")
stop2stn = (
    stns[["feed", "stop", "fua"]].reset_index()
    .explode(["feed", "stop"], ignore_index=True)
    .astype({"feed": I16, "stop": I32, "stn": UI16})
)#.view()

jrn_ic = (
    stopseq_ic.explode("stop", ignore_index=True)
    .merge(stop2stn, on=["feed", "stop"])
    .groupby(["feed", "stopseq"])
    ["fua"].nunique().rename("n_fuas")
    .reset_index()
    .query("n_fuas > 1")
    .drop(columns="n_fuas")
    .merge(jrn.reset_index(), on=["feed", "stopseq"])
    .set_index("jrn")
)#.view()

#%% Stops around stations
C.log("Mapping nearby stops to stations for intracity routing")
stn_stops = (
    stns.to_crs(C.CRS_EU)
    .buffer(params.STATION_BUFFER_RADIUS)
    .rename("geometry").reset_index()
    .to_crs(C.CRS_DEG)
    .sjoin(stops, predicate="contains")
    [["stn", "fua", "feed", "stop"]]
    .astype({"stn": UI16})
    .reset_index(drop=True)
)
stn_stops = (
    pd.concat([stop2stn, stn_stops])
    .drop_duplicates(ignore_index=True)
)#.view()

#%% Intracity station sequences [10s]
C.log("Identifying intracity station sequences")
n = stopseq["stop"].str.len().values # no. of stops in each stop seq
stnseq = (
    stopseq.explode("stop").astype({"stop": I32})
    .assign(stop_pos=lambda df: I16(
        np.arange(len(df)) - np.repeat(n.cumsum() - n, n)))
    .merge(stn_stops, on=["feed", "stop"])
    .drop_duplicates(["feed", "stopseq", "stn"], ignore_index=True)
)
imp_stopseq = (
    stnseq.groupby(["feed", "stopseq"])
    .size().rename("n_stns")
    .reset_index()
    .query("n_stns > 1")
    [["feed", "stopseq"]]
)
stnseq = (
    stnseq.merge(imp_stopseq, on=["feed", "stopseq"])
    .sort_values("stop_pos")
    .drop_duplicates(subset=["feed", "stopseq", "stop_pos"])
    .groupby(["feed", "stopseq"])
    [["stop_pos", "stn"]].agg(list)
    .pipe(lambda df: df[df["stn"].str.len() > 1])
    .reset_index()
)#.view()

#%% Timetable [26s]
C.log("Processing interstation timetable")
tt = (
    jrn.reset_index().astype({"jrn": I32})
    [["jrn", "feed", "trip", "timeseq", "stopseq", "start", "tz_gap"]]
    .merge(stnseq, on=["feed", "stopseq"])
    .merge(C.load("gtfs/db-time_seq")
           .rename(columns={"id": "timeseq", "fid": "feed", "arr_time": "arr"}),
           on=["feed", "timeseq"])
    .drop(columns=["feed", "trip", "timeseq", "stopseq"])
    .assign(arr=lambda df: [np.take(a, i) for a, i in zip(df["arr"], df["stop_pos"])])
    .assign(wait=lambda df: [np.take(a, i) for a, i in zip(df["wait"], df["stop_pos"])])
    .explode(["stn", "arr", "wait"], ignore_index=True)
    .astype({"stn": UI16, "arr": I32, "wait": I16})
    .query("arr >= 0")
)
tt["arr"] += (tt["start"] - tt["tz_gap"].astype(I32) * 3600) # convert to GMT
tt["dep"] = I16((tt["arr"] + tt["wait"]) / 60) # convert to minutes
tt["arr"] = I16((tt["arr"] / 60)) # convert to minutes
tt = tt.query("arr <= dep")
multistn_jrn = tt["jrn"].value_counts().pipe(lambda x: x[x > 1]).index
tt = tt[tt["jrn"].isin(multistn_jrn)]
tt = tt.sort_values(["jrn", "arr"], ignore_index=True)
tt = tt[["jrn", "stn", "arr", "dep"]]#.view()

#%% Lines [30s]
C.log("Identifying lines from timetable station sequences")
lines = (
    tt.groupby("jrn")["stn"].agg(tuple).reset_index()
    .merge(jrn[["agency", "tz_gap", "rail"]], on="jrn")
    .groupby(["agency", "rail", "tz_gap", "stn"])
    ["jrn"].agg(list).reset_index()
    .astype({"agency": str})
    .rename_axis("line")
    .reset_index().astype({"line": I32})
)
lines["intercity"] = (
    lines.set_index("line")
    ["stn"].explode().astype(int)
    .map(stns["fua"]).rename("fua")
    .reset_index().drop_duplicates()
    ["line"].value_counts().sort_index()
) > 1

#%% Major stations [6s]
C.log("Filtering major stations based on the city pairs they contribute to")
# No. of city OD pairs which are connected by lines that pass through `stn`
## Station pairs supporting each mode-specific city OD pair
adj = defaultdict(set)
stn_od = defaultdict(set)
all_od = set()
for line, r in lines[["rail", "stn"]].iterrows():
    is_rail = bool(r["rail"])
    stn_seq = tuple(dict.fromkeys(map(int, r["stn"])))
    stn_fua = {stn: int(stns.at[stn, "fua"]) for stn in stn_seq}
    fua_seq = tuple(dict.fromkeys(stn_fua.values()))
    line_od = {(is_rail, *sorted(x)) for x in combinations(fua_seq, 2)}
    all_od.update(line_od)
    for stn in stn_seq:
        stn_od[stn].update(line_od)
    for src, trg in combinations(stn_seq, 2):
        if stn_fua[src] != stn_fua[trg]:
            od = (is_rail, *sorted((stn_fua[src], stn_fua[trg])))
            adj[src].add((trg, od))
            adj[trg].add((src, od))
## Remove a station only when every OD retains another station-pair witness
selected_stns = set(stns.index)
od_support = Counter(od for src in selected_stns for 
                     trg, od in adj[src] if src < trg)
stn_order = sorted(selected_stns, key=lambda x: (len(stn_od[x]), x))
for stn in stn_order:
    removed = Counter(od for trg, od in adj[stn] if trg in selected_stns)
    if all(od_support[od] > n for od, n in removed.items()):
        selected_stns.remove(stn)
        od_support.subtract(removed)
stns2 = stns.loc[sorted(selected_stns)]#.view()

#%% Manually fix station names
# Export all candidate names with city & coordinates
candidate_names = (
    stns2.assign(first_name=stns2["name"].str.split(" | ").str[0])
    .reset_index().rename(columns={"stn": "stn_id", "name": "full_name"})
    .merge(fuas[["fua", "name"]].rename(columns={"name": "city"}), on="fua")
    .set_index(["stn_id", "first_name", "full_name", "city"])
    .get_coordinates().set_axis(["lon", "lat"], axis=1)
)
candidate_names.to_csv(C.DATA / "gtfs/stn-names-default.csv")
# Update station names with manually corrected ones
stns2 = (
    stns2.drop(columns="name", errors="ignore")
    .merge(pd.read_csv(C.DATA / "gtfs/stn-names-revised.csv")
           .rename(columns={"stn_id": "stn"})
           [["stn", "name"]], on="stn")
    .set_index("stn")
)#.view()

#%% Update lines
lines2 = (
    lines.set_index("line")
    ["stn"].explode().astype(int).reset_index()
    .merge(stns2["fua"], on="stn")
    .groupby("line")["stn"].agg(tuple)
    .pipe(lambda x: x[x.apply(len) > 1])
    .reset_index()
    .merge(lines.drop(columns="stn"), on="line")
)#.view()

#%% Segments (without geometry)
C.log("Identifying consecutive interstation segments")
seg = (
    lines2.reset_index().astype({"line": I32})
    .groupby("stn")["line"].agg(set).reset_index()
)
seg["od"] = seg.pop("stn").apply(lambda x: list(zip(x[:-1], x[1:])))
seg = seg.explode("od", ignore_index=True)
seg = seg.groupby("od")["line"].agg(lambda x: set().union(*x))
seg = seg.reset_index().rename_axis("seg")
seg["src"], seg["trg"] = list(zip(*seg.pop("od")))
seg = seg.astype({"src": I32, "trg": I32})
seg["mode"] = (
    seg["line"].explode().reset_index()
    .merge(lines2[["line", "rail"]], on="line")
    .astype({"rail": int})
    .groupby("seg")["rail"]
    .agg(lambda x: tuple(sorted(set(x)))).reset_index()
    .set_index("seg")["rail"]
    .map({(0,): "Bus", (1,): "Rail", (0, 1): "Both"})
)
seg["intercity"] = seg["src"].map(stns["fua"]) != seg["trg"].map(stns["fua"])
seg = seg[["src", "trg", "mode", "intercity", "line"]]#.view()

#%% Update other tables [3s]
C.log("Updating other tables")
jrn2 = (
    lines2[["line", "jrn"]]
    .explode("jrn").astype({"jrn": I32})
    .merge(jrn, on="jrn")
    .astype({"line": I32})
    .merge(tt.groupby("jrn")[["jrn", "dep"]].head(1), on="jrn")
    [["jrn", "line", "dateset", "dep", "feed", "trip", "stopseq", "timeseq"]]
    .astype({"jrn": I32})
)#.view()
tt2 = (
    tt[(tt["stn"].isin(stns2.index)) &
       (tt["jrn"].isin(jrn2["jrn"]))]
)#.view()
datesets2 = datesets.loc[jrn2["dateset"].unique()].sort_index()
datesets2.columns = list(map(str, datesets2.columns))
datesets2 = datesets2.reset_index().astype({"dateset": I32})#.view()

#%% Export [3s]
C.log("Saving intercity data tables")
for df, table in [
    (stns2.reset_index().astype({"stn": UI16}), "ic-stations"),
    (lines2.drop(columns="jrn"), "ic-lines"),
    (jrn2, "ic-journeys"),
    (datesets2, "ic-datesets"),
    (tt2, "ic-timetable"),
    (seg, "ic-segments"),
]:
    C.log(f"Table `{table}`: {len(df):,} rows")
    # print("Previously", table, len(C.load(table)))
    C.save(df, table)
