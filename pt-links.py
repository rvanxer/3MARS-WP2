"""Public transport (PT) links – intercity and intracity."""

#%% Imports
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from scipy import sparse

import config as C

UI16, F32 = np.uint16, np.float32

#%% Load data
stns = C.load("ic-stations")[["stn", "node_id", "fua"]]
lines = C.load("ic-lines", columns=["line", "operator", "rail", "stn"])
jrn = C.load("ic-journeys", columns=["jrn", "line", "dateset"])
tt = C.load("ic-timetable", columns=["jrn", "stn", "arr", "dep"])
seg = C.load("ic-segments").drop(columns="geometry")

#%% P-space station pairs by line
stn_pairs = []
seg_length = dict(seg.groupby(["src", "trg", "mode"])["len_km"].median())
line_od = lines.assign(mode=lines["rail"].map({False: "Bus", True: "Rail"}))
for stn_seq, mode in zip(line_od["stn"], line_od["mode"]):
    dist = np.array([
        seg_length.get((src, trg, mode), np.nan)
        for src, trg in zip(stn_seq[:-1], stn_seq[1:])
    ])
    cum_dist = np.r_[0, np.cumsum(np.nan_to_num(dist))]
    cum_missing = np.r_[0, np.cumsum(np.isnan(dist))]
    stn_pairs.append([
        (stn_seq[i], stn_seq[j],
         cum_dist[j] - cum_dist[i]
         if cum_missing[i] == cum_missing[j] else np.nan)
        for i, j in combinations(range(len(stn_seq)), 2)
        if stn_seq[i] != stn_seq[j]
    ])
line_od = lines.assign(od=stn_pairs).explode("od", ignore_index=True)
line_od[["src", "trg", "dist"]] = pd.DataFrame(line_od.pop("od").tolist())
line_od = line_od.drop_duplicates(["line", "src", "trg"], ignore_index=True)
line_od = line_od.query("operator != '.Local'").reset_index(drop=True)#.view()

#%% Links from line station pairs
links = (
    line_od.groupby(["src", "trg", "rail", "operator"])
    .agg({"line": "count", "dist": "median"})
    .rename(columns={"line": "n_lines"})
    .reset_index()
    .merge(stns[["stn", "fua"]].set_axis(["src", "src_fua"], axis=1))
    .merge(stns[["stn", "fua"]].set_axis(["trg", "trg_fua"], axis=1))
    .assign(intercity=lambda df: df.pop("src_fua") != df.pop("trg_fua"))
    .rename_axis("link")
)#.view()
link2line = (
    links.reset_index()
    .merge(line_od, on=["src", "trg", "rail", "operator"])
    [["link", "line", "src", "trg"]]
)#.view()

#%% P-space timetable [8s]
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

tt2 = tt.sort_values(["jrn", "arr"], ignore_index=True)
tt2 = pd.DataFrame(dict(zip(
    ["jrn", "src", "trg", "dep", "arr"],
    get_timetable_pairs(
        tt2["jrn"].values, tt2["stn"].values,
        tt2["arr"].values, tt2["dep"].values)
))).merge(jrn, on="jrn")
tt2 = tt2.merge(
    lines.query("operator != '.Local'")
    [["line", "operator", "rail"]]
    .astype({"operator": "category"}), on="line"
).drop(columns="line")
for col in ["src", "trg"]:
    tt2[col] = (pd.Categorical(tt2[col], categories=stns["stn"])
                .rename_categories(dict(zip(stns["stn"], stns["node_id"]))))
tt2["mode"] = tt2.pop("rail").map(
    {True: "R", False: "B"}).astype("category")

## Assign dates
dates = C.load("ic-datesets").set_index("dateset")
offsets = (pd.to_datetime(dates.columns, format="%Y%m%d") 
           - pd.Timestamp(C.PARAMS["SERVICE_START"])).days.values
rows, cols = np.nonzero(dates.values)
dates = pd.DataFrame({"dateset": dates.index[rows], "date": cols})
dates["date"] = np.int16(offsets[cols])
tt2 = tt2.merge(dates, on="dateset")[[
    "mode", "operator", "src", "trg", "dep", "arr", "date"
]].view()
del dates

C.save(tt2, "ic-pt-timetable")

#%% Graph links [7s]
links = (
    tt2.assign(time=tt2["arr"] - tt2["dep"])
    .groupby(["mode", "operator", "src", "trg"], observed=True)
    ["time"].agg(["count", "mean", "median", "std"])
    .set_axis(["nservices", "avg_time", "median_time", "std_time"], axis=1)
    .astype(np.float32).astype({"nservices": np.int32})
    .reset_index()
)
total_days = (C.PARAMS["SERVICE_END"] - C.PARAMS["SERVICE_START"]).days + 1
links["avg_freq"] = np.float32(links["nservices"] / total_days)
links.view();

C.save(links, "ic-pt-links")

#%%
