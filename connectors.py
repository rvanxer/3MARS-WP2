"""Compute access travel times for connector links. [21:43]"""

#%% Imports
import numpy as np
import pandas as pd

import config as C
from osrm import get_travel_times

#%% Load FUAs
fuas = (
    C.load("fuas")
    .rename(columns={"id": "fua"})
    .set_index("fua")
)#.view()

#%% Population grid (origins) [6s]
popu = (
    C.load("popu-grid")
    .pipe(C.pdf2gdf, "x", "y", C.CRS_EU)
    .rename_axis("cell").reset_index()
    .sjoin(fuas[["geometry"]].to_crs(C.CRS_EU))
    .astype({"cell": np.int32, "fua": np.int16})
    .set_index(["cell", "fua", "popu"])
    .to_crs(C.CRS_DEG)
    .get_coordinates().astype(np.float32)
    .reset_index()
)#.view()

#%% Transport hubs (destinations)
hubs = (
    pd.concat([
        C.load("airports")
        .rename(columns={"iata": "hub"})
        [["hub", "name", "geometry"]]
        .assign(kind="Airport"),
        
        C.load("ic-stations")
        .rename(columns={"stn": "hub"})
        [["hub", "name", "geometry"]]
        .assign(kind="Station")
    ]).astype({"hub": str})
    .sjoin(fuas[["geometry"]])
    .set_index(["hub", "kind", "name", "fua"])
    .get_coordinates().astype(np.float32)
    .reset_index()
)#.view()

#%% Compute access times
def get_access_times(fua: int, popu=popu, hubs=hubs):
    osm_path = C.DATA / f"osm/city/{fua}.osm.pbf"
    src = popu.query(f"fua == {fua}").reset_index(drop=True)
    trg = hubs.query(f"fua == {fua}").reset_index(drop=True)
    ttm = get_travel_times(src, trg, osm_path)
    if ttm is not None:
        return (
            ttm.astype({"time": np.int32, "dist": np.int32})
            .merge(src[["cell", "popu"]], left_on="src_id", right_index=True)
            .merge(trg[["hub", "kind"]], left_on="trg_id", right_index=True)
            .assign(fua=fua).astype({"hub": str})
            [["fua", "cell", "popu", "hub", "kind", "dist", "time"]]
        )

# x = get_access_times(7); x # Amsterdam
# x = get_access_times(1); x # Aalborg
# x = get_access_times(254); x # Paris

#%% Run for all cities [21:38]
for fua, fua_name in zip(fuas.index, fuas["name"]):
    try:
        fpath = C.mkdir(C.DATA / "connectors") / f"{fua}.parquet"
        if not fpath.exists():
            C.log(f"Processing #{fua}: {fua_name}")
            ttm = get_access_times(fua)
            ttm.to_parquet(fpath)
    except Exception as e:
        C.error(f"{fua}: {fua_name}: {e}")

#%% Weighted travel times
connectors = []
for fua_id in fuas.index:
    df = C.load(f"connectors/{fua_id}", quiet=True)
    if df is not None:
        df["time"] *= (df["popu"] / 60) # also convert to minutes
        df = df.groupby("hub")[["time", "dist", "popu"]].sum()
        df["dist"] /= df["popu"]
        df["time"] /= df.pop("popu")
        df = df.astype({"dist": np.float32, "time": np.float32})
        df = hubs[["fua", "hub", "kind", "name"]].merge(df, on="hub")
        connectors.append(df)
connectors = pd.concat(connectors, ignore_index=True)#.view(1)
C.save(connectors, "connectors")
