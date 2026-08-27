"""Intercity car travel times."""

#%% Imports
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import LineString

import config as C
from osrm import get_travel_times

#%% OD points: FUA centroids
fuas = (
    C.load("fuas")
    [["id", "name", "icc", "centre"]]
    .set_axis(["fua", "city", "icc", "geometry"], axis=1)
    .set_index("fua")
    .pipe(gpd.GeoDataFrame, crs=C.CRS_DEG)
).view()
pts = fuas.get_coordinates()

#%% Times using base highway network [1m27s]
C.log("Computing intercity driving times using base highway network")
osm_path = C.DATA / "osm/highways.osm.pbf"
ttm = get_travel_times(
    pts, pts, 
    osm_path,
    workdir=C.DATA / "osrm",
    server_start_timeout=900
).view()

#%% Manually add paths across intercity bridges
# directional distances (km) & times (min) between nearest bridge cities [Google Maps]
ttm_bridge = pd.DataFrame()
for fua_a, fua_b, dist_a2b, time_a2b, dist_b2a, time_b2a in [
    # UK-France connection across the English Channel
    ("Medway", "Dunkerque", 178, 164, 183, 161),
    # Sicily-mainland Italy connection
    ("Messina", "Reggio di Calabria", 24, 58, 24, 49),
    # Estonia-Finland connection across the Gulf of Finland
    ("Helsinki", "Tallinn", 180, 87.7, 182, 88.3),
]:
    fid_a = fuas[fuas["city"] == fua_a].index[0] # FUA ID of A
    fid_b = fuas[fuas["city"] == fua_b].index[0] # FUA ID of B
    # times from city A island to city B island
    a2b = pd.merge(
        ttm[ttm["src_id"] == fid_b].drop(columns="src_id").assign(_=True),
        ttm[ttm["trg_id"] == fid_a].drop(columns="trg_id").assign(_=True),
        how="outer", on="_", suffixes=["_b", "_a"]
    ).drop(columns="_")
    a2b["dist"] = a2b.pop("dist_a") + a2b.pop("dist_b") + dist_a2b
    a2b["time"] = a2b.pop("time_a") + a2b.pop("time_b") + time_a2b
    # times from city B island to city A island
    b2a = pd.merge(
        ttm[ttm["src_id"] == fid_a].drop(columns="src_id").assign(_=True),
        ttm[ttm["trg_id"] == fid_b].drop(columns="trg_id").assign(_=True),
        how="outer", on="_", suffixes=["_a", "_b"]
    ).drop(columns="_")
    b2a["dist"] = b2a.pop("dist_a") + b2a.pop("dist_b") + dist_b2a
    b2a["time"] = b2a.pop("time_a") + b2a.pop("time_b") + time_b2a
    # combine paths
    ttm_bridge = pd.concat([ttm_bridge, a2b, b2a], ignore_index=True)
ttm2 = (
    pd.concat([ttm, ttm_bridge], ignore_index=True)
    .rename(columns={"src_id": "src_fua", "trg_id": "trg_fua"})
    .astype({"src_fua": np.int16, "trg_fua": np.int16})
).view()
ttm2["time"] /= 60 # convert to minutes
ttm2["dist"] /= 1000 # convert to kilometres

C.save(ttm2, "car-links")

#%% Visualise
# max_hrs = 12
# df = ttm2.assign(time_h=ttm2["time"] / 60)
# df = df[df["time_h"] <= max_hrs]
# df["domestic"] = df["src_id"].map(fuas["icc"]) == df["trg_id"].map(fuas["icc"])
# df["geometry"] = [LineString(x) for x in zip(
#     df["src_id"].map(fuas.geometry), df["trg_id"].map(fuas.geometry))]
# df = gpd.GeoDataFrame(df, crs=C.CRS_DEG).to_crs(C.CRS_EU)
# _, ax = plt.subplots(figsize=(15, 10))
# ax.axis("off")
# ax.set_title(f"City OD connections less than {max_hrs} hours by drive")
# df.query("domestic").plot(ax=ax, color="b", alpha=0.3, lw=0.2, zorder=3)
# # df.query("~domestic").plot(ax=ax, color="r", alpha=0.2, lw=0.1, zorder=2)
# fuas.to_crs(C.CRS_EU).plot(ax=ax, markersize=5, color="k", zorder=5)
# cntr = C.load("countries").to_crs(C.CRS_EU)
# cntr.to_crs(C.CRS_EU).plot(ax=ax, fc="lightgrey", ec="w", lw=1, zorder=1)
# ax.legend(handles=[
#     mpl.patches.Patch(label=label, fc=color) for label, color in 
#     [("Domestic", "b"), ("International", "r")]
# ])
# cntr.to_crs(C.CRS_EU).plot(ax=ax, fc="none", ec="w", lw=0.8, zorder=4)
# x0, y0, x1, y1 = fuas.to_crs(C.CRS_EU).total_bounds
# ax.set_xlim(x0 - 1e5, x1 + 1e5)
# ax.set_ylim(y0 - 1e5, y1 + 1e5);

#%%
C.load("air-links")

