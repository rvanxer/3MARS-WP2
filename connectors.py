"""Compute access travel times for connector links."""

#%% Imports
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import subprocess
from shapely import LineString
from tqdm import tqdm

import config as C
from osrm import routing_session

#%% Load FUAs and countries
cntr = C.load("countries")[["icc", "geometry"]]#.view()

fuas = C.load("fuas").set_index("id").rename_axis("fua")#.view()

#%% Transport hubs (destinations)
# one row per airport-FUA or station-FUA pair
hubs = (
    pd.concat([
        (C.load("ic-stations")
         [["stn", "name", "geometry"]]
         .to_crs(fuas.crs)
         .sjoin(fuas[["geometry"]])
         .assign(hub=lambda df: [f"STN_{x:04d}" for x in df["stn"]])
         .assign(kind="Station")),
        (C.load("airports")
         [["iata", "name", "fua", "geometry"]]
         .explode("fua").astype({"fua": int})
         .assign(hub=lambda df: ["AIR_" + x for x in df["iata"]])
         .assign(kind="Airport"))
    ], ignore_index=True)
    .merge(fuas["icc"], on="fua")
    [["hub", "kind", "name", "fua", "icc", "geometry"]]
)
hubs = pd.concat([hubs, hubs.get_coordinates()], axis=1)#.view(1)

#%% FUA boundaries by country, inc. buffers of external airports
hub2fua_buffers = (
    hubs.merge(fuas["centre"], on="fua")
    .assign(geometry=lambda df: [
        LineString([r.geometry, r["centre"]])
        for _, r in df.iterrows()
    ]).to_crs(C.CRS_EU)
    .set_index(["fua", "hub"])
    .buffer(C.PARAMS["OSM_BUFFER_RADIUS"] * 1000)
    .rename("geometry").reset_index()
    .dissolve("fua")[["geometry"]]
    .to_crs(C.CRS_DEG)
)
fua_parts = (
    pd.concat([hub2fua_buffers, fuas[["geometry"]]])
    .dissolve("fua").reset_index()
    .overlay(cntr, how="intersection")
)#.view()

#%% OSM file of combined buffered FUA boundaries [2:26]
buf_file = C.DATA / "osm/EU-FUA-buffers.osm.pbf"
if not buf_file.exists():
    tmp = C.mkdir(C.DATA / "osm/buffers-icc")
    snapshot_date = C.PARAMS["OSM_SNAPSHOT_DATE"].strftime("%y%m%d")
    for icc, df in (pbar := tqdm(fua_parts.groupby("icc"))):
        pbar.set_description(icc)
        in_json = f"{tmp}/{icc}.json"
        df.dissolve()[["geometry"]].to_file(in_json)
        subprocess.run([
            "osmium", "extract", "--strategy=complete_ways",
            "-S", "relations=true", "--overwrite",
            "-p", in_json,
            "-o", f"{tmp}/{icc}.osm.pbf",
            f"{C.DATA}/osm/country-{snapshot_date}/{icc}.osm.pbf"
        ], check=True)
    subprocess.run([
        "osmium", "merge", "--no-progress", "--overwrite",
        "-o", str(buf_file), *map(str, tmp.glob("*.osm.pbf"))
    ], check=True)
    shutil.rmtree(tmp)

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

#%% Connector distances & times for one city
def get_connector_times(fua: int, hubs=hubs, cells=popu, osm_path=buf_file):
    cells = popu[popu["fua"] == fua].reset_index(drop=True)
    hubs = hubs[hubs["fua"] == fua].reset_index(drop=True)
    pts = pd.concat([hubs, cells], ignore_index=True)
    pts = C.pdf2gdf(pts[["x", "y"]], "x", "y", C.CRS_DEG)
    
    connectors = []
    with routing_session(
        osm_paths=osm_path,
        extent=pts.geometry,
        probe_pt=tuple(pts.geometry.iloc[0].coords[0]),
        buffer_radius=C.PARAMS["OSM_BUFFER_RADIUS"],
        workdir=C.DATA / "osrm",
    ) as table:
        for kind, src, trg, src_col, trg_col in [
            ("Access", cells, hubs, "fua", "hub"),
            ("Egress", hubs, cells, "hub", "fua"),
            ("Interhub", hubs, hubs, "hub_x", "hub_y")
        ]:
            if (df := table(src, trg)) is None:
                continue
            df = df.merge(src, left_on="src_id", right_index=True)
            df = df.merge(trg, left_on="trg_id", right_index=True)
            df = df.dropna(subset=["dist", "time"])
            if "popu" in df.columns:
                df["dist"] *= df["popu"]; df["time"] *= df["popu"]
                df = df.groupby("hub")[["popu", "dist", "time"]].sum()
                df["dist"] /= df["popu"]; df["time"] /= df["popu"]
                df = df.reset_index()
                df["fua"] = f"FUA_{fua:03d}"
            df["dist"] /= 1000 # convert to km
            df["time"] /= 60 # convert to minutes
            df = df.rename(columns={src_col: "src", trg_col: "trg"})
            df = df[["src", "trg", "dist", "time"]]
            df = df[df["src"] != df["trg"]]
            df.insert(0, "kind", kind)
            connectors.append(df)
    return pd.concat(connectors, ignore_index=True)

con = SimpleNamespace()
# con.aberdeen = get_connector_times(2).view() # 13s
# con.amsterdam = get_connector_times(7).view() # 43s
# con.athens = get_connector_times(15).view() # 16s
# con.paris = get_connector_times(254).view() # 50s

#%%
docker_raw = Path.home() /\
    "/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"

def record_storage(title):
    for label, command in [
        ("Containers", ["docker", "ps", "-a", "--size"]),
        ("data/osrm allocated KiB", ["du", "-sk", str(C.DATA / "osrm")]),
        ("Docker.raw allocated KiB", ["du", "-sk", str(docker_raw)]),
        ("Docker system usage", ["docker", "system", "df", "-v"]),
        ("Host free space", ["df", "-h", str(C.DATA)]),
    ]:
        C.log(f"DOCKER STORAGE for '{title}'")
        try:
            C.log(label)
            subprocess.run(command, text=True, check=False)
        except:
            pass

#%% Process all cities [3:14:16 (3 tries)]
#TODO: Improve performance, esp. disk usage in Docker
i = 0 # counter of processed FUA
record_storage("start")
for fua, r in fuas.iterrows():
    fpath = C.mkdir(C.DATA / "connectors") / f"{fua}.csv"
    if fpath.exists():
        continue
    try:
        C.log(f"Processing {fua}: {r['name']}")
        ttm = get_connector_times(fua)
        ttm.to_csv(fpath, index=False)
        i += 1
    except Exception as e:
        C.error(f"Couldn't process {fua}: {r['name']}: {e}")
        try:
            record_storage(f"failure: {fua}; completed={i}")
        except:
            pass
        continue
    if i % 10 == 0:
        record_storage(f"afer {i} cities; latest={fua}")

#%% Combine connectors
connectors = pd.concat([
    pd.read_csv(f).assign(fua=np.int16(f.stem))
    .astype({"dist": np.float32, "time": np.float32})
    for f in C.DATA.glob("connectors/*.csv")
]).sort_values("fua", ignore_index=True).view()
C.save(connectors, "connectors")
