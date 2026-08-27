#%% Imports
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

warnings.filterwarnings("ignore", category=pd.errors.ChainedAssignmentError)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas._config")

#%% Country-level rai/highway OSM geodatabase [25:41]
C.log("Downloading country-wise OSM geodatabase extracts")
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
outdir = C.mkdir(C.DATA / "osm/country")
for icc, uri in (pbar := tqdm(osm_uris.items())):
    if (outfile := outdir / f"{icc}.osm.pbf").exists():
        continue
    pbar.set_description(icc)
    full_pbf = outdir / f"full-{icc}.osm.pbf"
    url = f"https://download.geofabrik.de/europe/{uri}-latest.osm.pbf"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(full_pbf, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    cmd = (f"osmium tags-filter --overwrite -f pbf -o {outfile} {full_pbf} "
           "w/highway w/railway w/bridge w/tunnel "
           "r/route=train r/route=light_rail r/route=subway")
    subprocess.run(shlex.split(cmd))
    full_pbf.unlink()

#%% Railway network [1:53]
C.log("Generating railway network")
rail = C.load(filename := "osm/railways")
if rail is None:
    rail = []
    indir = C.DATA / "osm/country"
    outdir = C.mkdir(C.DATA / "osm/railways")
    for icc in (pbar := tqdm(osm_uris)):
        try:
            pbar.set_description(icc)
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

#%% Highway network [1:37]
C.log("Generating highway network")
hway = C.load(filename := "osm/highways")
if hway is None:
    ## Extract highways from country-wise OSM extracts
    tmpdir = C.mkdir(C.DATA / "tmp_highways")
    outdir = C.mkdir(C.DATA / "osm/highways")
    for icc in (pbar := tqdm(osm_uris)):
        pbar.set_description(icc)
        tmpfile = tmpdir / f"{icc}.osm.pbf"
        cmd = (f"osmium tags-filter --overwrite -f pbf -o {tmpfile} " + 
               f"{C.DATA}/osm/country/{icc}.osm.pbf " + 
               " ".join([f"w/highway=" + x for x in [
                    "motorway", "motorway_link",
                    "trunk", "trunk_link",
                    "primary", "primary_link"
                ]]))
        subprocess.run(shlex.split(cmd), check=True)
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

#%% Urban area group extracts by country [1:48]
fuas = C.load("fuas").to_crs(C.CRS_DEG).view()

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
        
# runtime 1:48
for icc, df in (pbar := tqdm(fuas.groupby("icc"))):
    pbar.set_description(icc)
    in_osm = C.DATA / f"osm/country/{icc}.osm.pbf"
    out_osm = C.mkdir(C.DATA / "osm/citygroup") / f"{icc}.osm.pbf"
    in_json = C.DATA / f"osm/citygroup/{icc}.geojson"
    if not in_json.exists():
        df[["geometry"]].dissolve().to_file(in_json, driver="GeoJSON")
    clip_osmium(in_osm, in_json, out_osm, overwrite=False)
    in_json.unlink()

#%% Urban area extracts [6:15]
for fua_id, r in (pbar := tqdm(fuas.iterrows(), total=len(fuas))):
    pbar.set_description(r["name"])
    in_osm = C.DATA / f"osm/citygroup/{r.icc}.osm.pbf"
    in_json = C.mkdir(C.DATA / "osm/city") / f"{fua_id}.geojson"
    if not in_json.exists():
        boundary = gpd.GeoDataFrame(r.to_frame().T, crs=C.CRS_DEG)
        boundary.to_file(in_json, driver="GeoJSON")
    out_osm = C.DATA / "osm/city" / f"{fua_id}.osm.pbf"
    clip_osmium(in_osm, in_json, out_osm, overwrite=0)
    in_json.unlink()
