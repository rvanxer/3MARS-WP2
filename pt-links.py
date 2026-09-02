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
stns = C.load("ic-stations")[["stn", "fua"]]
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
line_od.view(1);

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

#%% P-space link travel time
@njit
def get_timetable_pairs(jrn_id, stn, arr, dep):
    """Expand ordered journey stops into P-space timetable rows."""
    n_pairs = 0
    start = 0
    while start < len(jrn_id):
        end = start + 1
        while end < len(jrn_id) and jrn_id[end] == jrn_id[start]:
            end += 1
        n_stn = end - start
        n_pairs += n_stn * (n_stn - 1) // 2
        start = end
    out_jrn = np.empty(n_pairs, dtype=jrn_id.dtype)
    out_src = np.empty(n_pairs, dtype=stn.dtype)
    out_trg = np.empty(n_pairs, dtype=stn.dtype)
    out_time = np.empty(n_pairs, dtype=np.int32)
    k = 0
    start = 0
    while start < len(jrn_id):
        end = start + 1
        while end < len(jrn_id) and jrn_id[end] == jrn_id[start]:
            end += 1
        for i in range(start, end - 1):
            for j in range(i + 1, end):
                out_jrn[k] = jrn_id[start]
                out_src[k] = stn[i]
                out_trg[k] = stn[j]
                out_time[k] = arr[j] - dep[i]
                k += 1
        start = end
    return out_jrn, out_src, out_trg, out_time

tt2 = tt.sort_values(["jrn", "arr"], ignore_index=True)
tt_pairs = get_timetable_pairs(
    tt2["jrn"].to_numpy(), tt2["stn"].to_numpy(),
    tt2["arr"].to_numpy(), tt2["dep"].to_numpy(),
)
tt2 = pd.DataFrame(dict(zip(["jrn", "src", "trg", "time"], tt_pairs)))
links2 = (
    tt2.merge(jrn[["jrn", "line"]], on="jrn")
    .merge(link2line, on=["line", "src", "trg"])
    .groupby("link")
    ["time"].agg(time="median", sd_time=lambda x: x.std(ddof=0))
    .merge(links, on="link")
)#.view()

#%% Link frequency
# median daily frequency over the link's active service dates
# matrix multiplication gives journeys per link and date:
# link-line incidence x line-dateset journey counts x dateset-date activity
line_ids = pd.Index(lines["line"].drop_duplicates())
date_path = C.DATA / "ic-datesets.parquet"
dateset_ids = pd.Index(C.load(
    "ic-datesets", columns=["dateset"])["dateset"])
date_cols = [x for x in pq.ParquetFile(date_path).schema.names
             if x != "dateset"]
line2dateset = (
    jrn.groupby(["line", "dateset"])
    .size().rename("n_jrn").reset_index()
)
line_dateset_mat = sparse.coo_matrix(
    (UI16(line2dateset["n_jrn"]), (
        line_ids.get_indexer(line2dateset["line"]),
        dateset_ids.get_indexer(line2dateset["dateset"]),
    )), shape=(len(line_ids), len(dateset_ids)), dtype=UI16
).tocsr()
link_ids = pd.Index(links2.index)
active_link2line = link2line[link2line["link"].isin(link_ids)]
link_line_mat = sparse.coo_matrix(
    (np.ones(len(active_link2line), dtype=UI16), (
        link_ids.get_indexer(active_link2line["link"]),
        line_ids.get_indexer(active_link2line["line"]),
    )), shape=(len(links2), len(line_ids)), dtype=UI16
).tocsr()
link_dateset_mat = link_line_mat @ line_dateset_mat
freq = np.empty((len(links2), len(date_cols)), dtype=UI16)
for i in range(0, len(date_cols), 100):
    cols = date_cols[i: i + 100]
    freq[:, i: i + len(cols)] = link_dateset_mat @ UI16((
        C.load("ic-datesets", columns=["dateset"] + cols, quiet=True)
        .set_index("dateset").reindex(dateset_ids)
    )[cols])
links2["freq"] = F32([np.median(x[x > 0]) for x in freq])

#%% Export
links3 = (
    links2.reset_index(drop=True)
    [["src", "trg", "intercity", "rail", "operator", "n_lines",
      "dist", "time", "sd_time", "freq"]]
    .astype({"n_lines": UI16, "dist": F32, "time": F32,
             "sd_time": F32, "freq": F32})
)#.view(1)
C.save(links3, "pt-links")
