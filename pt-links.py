"""Public transport (PT) links – intercity and intracity."""

#%% Imports
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

import config as C

UI16, F32 = np.uint16, np.float32

#%% Load data
stns = C.load("ic-stations")[["stn", "fua"]]
lines = C.load("ic-lines", columns=["line", "operator", "rail", "stn"])
jrn = C.load("ic-journeys", columns=["jrn", "line", "dateset"])
tt = C.load("ic-timetable", columns=["jrn", "stn", "arr", "dep"])
seg = C.load("ic-segments").drop(columns="geometry")

#%% Segments (consecutive station pairs) by line
df = lines.assign(mode=lines["rail"].map({False: "Bus", True: "Rail"}))
df["od"] = [list(zip(x[:-1], x[1:])) for x in df["stn"]]
df = df.explode("od").dropna(subset="od", ignore_index=True)
df["src"], df["trg"] = list(zip(*df.pop("od")))
df = df.drop_duplicates(["line", "src", "trg"], ignore_index=True)
# mode-specific segment lengths
seg_dist = (seg.groupby(["src", "trg", "mode"])
            ["len_km"].median().rename("dist").reset_index())
line_seg = df.merge(seg_dist, "left", on=["src", "trg", "mode"]).view()

#%% Links from line segments
links = (
    line_seg.groupby(["src", "trg", "rail", "operator"])
    .agg({"line": "count", "dist": "median"})
    .rename(columns={"line": "n_lines"})
    .reset_index()
    .merge(stns[["stn", "fua"]].set_axis(["src", "src_fua"], axis=1))
    .merge(stns[["stn", "fua"]].set_axis(["trg", "trg_fua"], axis=1))
    .assign(intercity=lambda df: df.pop("src_fua") != df.pop("trg_fua"))
    .rename_axis("link")
).view()
link2line = (
    links.reset_index()
    .merge(line_seg, on=["src", "trg", "rail", "operator"])
    [["link", "line", "src", "trg"]]
).view()

#%% Link travel time [3s]
tt2 = tt.sort_values(["jrn", "arr"], ignore_index=True)
jrn_groups = tt2.groupby("jrn", sort=False)
tt2["trg"] = jrn_groups["stn"].shift(-1)
tt2["trg_arr"] = jrn_groups["arr"].shift(-1)
links2 = (
    tt2.dropna(subset=["trg"])
    .rename(columns={"stn": "src"})
    .assign(time=lambda df: df.pop("trg_arr") - df["dep"])
    .astype({"trg": int})
    .merge(jrn[["jrn", "line"]], on="jrn")
    .merge(link2line, on=["line", "src", "trg"])
    .groupby("link")
    ["time"].agg(time="median", sd_time=lambda x: x.std(ddof=0))
    .merge(links, on="link")
).view()

#%% Link frequency
# median daily frequency over the link's active service dates
# matrix multiplication gives journeys per link and date:
# link-line incidence x line-dateset journey counts x dateset-date activity
line_ids = pd.Index(lines["line"].drop_duplicates())
date_path = C.DATA / "ic-datesets.parquet"
dateset_ids = pd.Index(C.load("ic-datesets", 
                              columns=["dateset"])["dateset"])
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
link_line_mat = sparse.coo_matrix(
    (np.ones(len(link2line), dtype=UI16), (
        link2line["link"],
        line_ids.get_indexer(link2line["line"]),
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
links2 = (
    links2.reset_index(drop=True)
    [["src", "trg", "intercity", "rail", "operator", "n_lines",
      "dist", "time", "sd_time", "freq"]]
    .astype({"n_lines": UI16, "dist": F32, "time": F32,
             "sd_time": F32, "freq": F32})
).view()
C.save(links2, "pt-links")

#%%
