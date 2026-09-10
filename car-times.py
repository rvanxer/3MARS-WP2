"""Intercity car travel times."""

#%% Imports
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C
from osrm import get_travel_times

#%% OD points: FUA centroids
fuas = (
    C.load("fuas")
    [["id", "name", "icc", "centre"]]
    .set_axis(["fua", "city", "icc", "geometry"], axis=1)
    .set_index("fua")
    .pipe(gpd.GeoDataFrame, crs=C.CRS_DEG)
)#.view()
pts = fuas.get_coordinates()

#%% Times using base highway network [1m27s]
C.log("Computing intercity driving times using base highway network")
ttm = get_travel_times(
    pts, pts,
    C.DATA / "osm/highways.osm.pbf",
    workdir=C.DATA / "osrm",
    server_start_timeout=900
)#.view()

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
)#.view()
ttm2["time"] /= 60 # convert to minutes
ttm2["dist"] /= 1000 # convert to kilometres

C.save(ttm2, "car-links")
