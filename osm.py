#%% Imports
import logging
from pathlib import Path
import shlex
from shutil import rmtree
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM
import requests
import subprocess
from shapely import LineString
from tqdm import tqdm

import config as C

params = C.load_params()

warnings.filterwarnings("ignore", category=pd.errors.ChainedAssignmentError)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas._config")
logging.getLogger("pyogrio").setLevel(logging.WARNING)

#%% Country-level rai/highway OSM geodatabase [53m19s]
snapshot_date = params.OSM_SNAPSHOT_DATE
snapshot_str = snapshot_date.strftime("%y%m%d")
C.log(f"Downloading country-wise OSM snapshots on {snapshot_date}")
osm_uris = dict(
    AT = "austria",
    BE = "belgium",
    BG = "bulgaria",
    CH = "switzerland",
    CZ = "czech-republic",
    DE = "germany",
    DK = "denmark",
    EE = "estonia",
    EL = "greece",
    ES = "spain",
    FI = "finland",
    FR = "france",
    HR = "croatia",
    HU = "hungary",
    IE = "ireland-and-northern-ireland",
    IT = "italy",
    LT = "lithuania",
    LU = "luxembourg",
    LV = "latvia",
    NL = "netherlands",
    NO = "norway",
    PL = "poland",
    PT = "portugal",
    RO = "romania",
    SE = "sweden",
    SI = "slovenia",
    SK = "slovakia",
    UK = "united-kingdom",
)
outdir = C.mkdir(C.DATA / f"osm/country-{snapshot_str}")
for icc, uri in osm_uris.items():
    if (outfile := outdir / f"{icc}.osm.pbf").exists():
        continue
    try:
        C.log(f"Downloading OSM extract for {icc}")
        full_pbf = outdir / f"full-{icc}.osm.pbf"
        url = "https://download.geofabrik.de/europe/"
        url += f"{uri}-{snapshot_str[2:]}.osm.pbf"
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(full_pbf, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        C.log(f"Extracting OSM extract for {icc}")
        subprocess.run(shlex.split(
            "osmium tags-filter --overwrite -f pbf "
            f"-o {outfile} {full_pbf} "
            "w/highway w/railway w/bridge w/tunnel "
            "r/route=train r/route=light_rail r/route=subway"
        ), check=True)
        full_pbf.unlink()
    except Exception as e:
        C.error(f"{icc}: {e}")

#%% Railway network [1m53s]
C.log("Generating railway network")
rail = C.load(filename := "osm/railways")
if rail is None:
    rail = []
    indir = C.DATA / f"osm/country-{snapshot_str}"
    outdir = C.mkdir(C.DATA / "osm/railways")
    for icc in osm_uris:
        try:
            cmd = (f"osmium tags-filter {indir}/{icc}.osm.pbf "
                   "r/route=train r/route=light_rail r/route=subway "
                   "w/railway=rail w/railway=light_rail "
                   "w/railway=subway w/railway=tram "
                   f"-o {outdir}/{icc}.osm.pbf --overwrite")
            subprocess.run(shlex.split(cmd))
            osm = OSM(f"{outdir}/{icc}.osm.pbf")
            df = osm.get_data_by_custom_criteria(
                custom_filter={"railway": ["rail"]},
                osm_keys_to_keep=None).geometry.dropna().reset_index()
            df = df[df.geom_type.isin(["LineString", "MultiLineString"])]
            df["len_km"] = df.to_crs(C.CRS_EU).length / 1000
            df.geometry = df.geometry.simplify(0.0001) # round off to ≈6-10 m
            df = df.assign(icc=icc)[["icc", "len_km", "geometry"]]
            rail.append(df)
        except Exception as e:
            C.log(f"{icc}: {e}", "error")
    rail = (pd.concat(rail, ignore_index=True)
            .astype({"icc": "category", "len_km": np.float32}))
    ## Manually add some connector links
    for icc, x1, y1, x2, y2, left, bottom, right, top in [
        # Sicily-mainland Italy connection
        ["IT", 15.560918, 38.210380, 15.633311, 38.221237,
        15.392392, 38.121901, 15.708970, 38.283881],
    ]:
        pts = (
            rail.cx[left: right, bottom: top]
            .rename_axis("way_id")
            .get_coordinates().reset_index()
        )
        groups = pts.groupby("way_id", sort=False)
        endpts = (
            pd.concat([groups.first(), groups.last()])
            .reset_index()
            .groupby(["x", "y"])
            ["way_id"].agg(list)
        )
        endpts = endpts[endpts.apply(len) > 1].reset_index()
        p1 = pts.loc[((pts["x"] - x1) ** 2 + (pts["y"] - y1) ** 2).argmin()]
        p2 = pts.loc[((pts["x"] - x2) ** 2 + (pts["y"] - y2) ** 2).argmin()]
        df = gpd.GeoDataFrame({"geometry": [
            LineString([(p1.x, p1.y), (p2.x, p2.y)]),
            LineString([(p2.x, p2.y), (p1.x, p1.y)])
        ]}, crs=C.CRS_DEG)
        df["len_km"] = df.to_crs(C.CRS_EU).length / 1000
        rail = pd.concat([rail, df.assign(icc=icc)], ignore_index=True)
    C.save(rail, filename, compression="zstd")

#%% Highway network [1m37s]
C.log("Generating highway network")
hway = C.load(filename := "osm/highways")
if hway is None:
    ## Extract highways from country-wise OSM extracts
    tmpdir = C.mkdir(C.DATA / "tmp_highways")
    outdir = C.mkdir(C.DATA / "osm/highways")
    for icc in (pbar := tqdm(osm_uris)):
        pbar.set_description(icc)
        tmpfile = tmpdir / f"{icc}.osm.pbf"
        subprocess.run(shlex.split(
            f"osmium tags-filter --overwrite -f pbf -o {tmpfile} " + 
            f"{C.DATA}/osm/country-{snapshot_str}/{icc}.osm.pbf " + 
            " ".join([f"w/highway=" + x for x in [
                "motorway", "motorway_link",
                "trunk", "trunk_link",
                "primary", "primary_link"
        ]])), check=True)
        outfile = outdir / tmpfile.name
        cmd = f"osmium cat -c version -o {outfile} --overwrite {tmpfile}"
        subprocess.run(shlex.split(cmd), check=True)
    rmtree(tmpdir)
    # Merge all networks
    osm_path = C.DATA / "osm/highways.osm.pbf"
    cmd = f"osmium merge --overwrite -o {osm_path} -f pbf "
    cmd += " ".join([str(f) for f in C.DATA.glob("osm/highways/*.osm.pbf")])
    subprocess.run(shlex.split(cmd), check=True)
    # Convert OSM PBF to GPKG
    gpkg_path = C.DATA / "osm/highways.gpkg"
    cmd = ("ogr2ogr --config OSM_USE_CUSTOM_INDEXING NO -f GPKG " +
           f"{gpkg_path} {osm_path} ").split()
    cmd += ["-sql", "SELECT * FROM lines WHERE highway IS NOT NULL"]
    cmd += "-nln roads -overwrite -lco SPATIAL_INDEX=YES".split()
    subprocess.run(cmd, check=True)
    
    ## Manually add some connector links
    hway = gpd.read_file(gpkg_path, columns=["geometry"])
    for x1, y1, x2, y2, left, bottom, right, top in [
        # UK-France connection across the English Channel
        [1.280245, 51.110284, 1.732929, 50.938103,
         1.188655, 50.875993, 1.916286, 51.146843],
        # Sicily-mainland Italy connection
        [15.560918, 38.210380, 15.633311, 38.221237,
         15.392392, 38.121901, 15.708970, 38.283881],
        # Finland-Estonia connection across Gulf of Finland
        [24.955124, 60.167073, 24.760075, 59.443544,
         24.404081, 59.203508, 25.476579, 60.379326]
    ]:
        pts = (
            hway.cx[left: right, bottom: top]
            .rename_axis("way_id")
            .get_coordinates().reset_index()
        )
        groups = pts.groupby("way_id", sort=False)
        endpts = (
            pd.concat([groups.first(), groups.last()])
            .reset_index()
            .groupby(["x", "y"])
            ["way_id"].agg(list)
        )
        endpts = endpts[endpts.apply(len) > 1].reset_index()
        p1 = pts.loc[((pts["x"] - x1) ** 2 + (pts["y"] - y1) ** 2).argmin()]
        p2 = pts.loc[((pts["x"] - x2) ** 2 + (pts["y"] - y2) ** 2).argmin()]
        df = gpd.GeoDataFrame({"geometry": [
            LineString([(p1.x, p1.y), (p2.x, p2.y)]),
            LineString([(p2.x, p2.y), (p1.x, p1.y)])
        ]}, crs=C.CRS_DEG)
        hway = pd.concat([hway, df], ignore_index=True)
    hway = hway.dissolve().explode().reset_index(drop=True)
    hway["len_km"] = hway.to_crs(C.CRS_EU).length / 1000
    hway = hway.astype({"len_km": np.float32})
    # export data
    gpkg_path.unlink()
    C.save(hway, filename, compression="zstd")

#%% Urban area extracts [8m3s]
fuas = C.load("fuas").to_crs(C.CRS_DEG)#.view()

def clip_osmium(in_osm, in_json, out_osm, overwrite=False):
    if out_osm.exists() and not overwrite:
        return
    try:
        C.mkdir(Path(out_osm).parent)
        cmd = ("osmium extract --strategy=complete_ways -S relations=true "
               f"--overwrite -p {in_json} -o {out_osm} {in_osm}")
        subprocess.run(cmd.split(), check=True)
    except Exception as e:
        print(f"ERROR in {str(in_json).split("/")[-1]}: {e}")
        
for icc, df in (pbar := tqdm(fuas.groupby("icc"))):
    pbar.set_description(icc)
    cntr_osm = C.DATA / f"osm/country-{snapshot_str}/{icc}.osm.pbf"
    grp_osm = C.mkdir(C.DATA / "osm/citygroup") / f"{icc}.osm.pbf"
    grp_json = C.DATA / f"osm/citygroup/{icc}.geojson"
    if not grp_json.exists():
        df[["geometry"]].dissolve().to_file(grp_json, driver="GeoJSON")
    clip_osmium(cntr_osm, grp_json, grp_osm, overwrite=False)
    for fua_id, r in df.iterrows():
        city_osm = C.DATA / "osm/city" / f"{fua_id}.osm.pbf"
        if not city_osm.exists():
            city_json = C.mkdir(C.DATA / "osm/city") / f"{fua_id}.geojson"
            if not city_json.exists():
                boundary = gpd.GeoDataFrame(r.to_frame().T, crs=C.CRS_DEG)
                boundary.to_file(city_json, driver="GeoJSON")
            clip_osmium(grp_osm, city_json, city_osm, overwrite=False)
            city_json.unlink()
    grp_json.unlink()
    grp_osm.unlink()
grp_osm.parent.unlink()
