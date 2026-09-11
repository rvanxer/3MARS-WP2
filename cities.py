#%% Imports
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

#%% FUA boundaries: Europe – JRC
"""FUA boundaries for most study countries come from the EU Joint Research 
Centre (JRC)."""
C.log("Reading JRC FUA boundaries")
fua_jrc = (
    gpd.read_file(C.URLS["fua-jrc"])
    .rename(columns={"FUA_NAME": "name", "COUNTRY_CO": "icc"})
    .drop_duplicates(["icc", "name"])
    .to_crs(C.CRS_DEG)
    [["name", "icc", "geometry"]]
)

#%% FUA boundaries: CH & NO – GISCO
"""For the special case of Switzerland (CH) and Norway (NO), the boundaries 
are obtained from the Geographic Information System of the EU Commission 
(GISCO) portal."""
C.log("Reading GISCO FUA boundaries")
fua_gisco = (
    gpd.read_file(C.URLS["fua-gisco"],
                  columns=["CNTR_CODE", "URAU_NAME", "geometry"])
    .query("CNTR_CODE in ('CH', 'NO')")
    .rename(columns={"CNTR_CODE": "icc"})
    .to_crs(C.CRS_DEG)
)
fua_gisco["name"] = fua_gisco["URAU_NAME"].str.replace("FUA ", "")
fua_gisco = fua_gisco[["icc", "name", "geometry"]]

#%% Combine FUAs, filter for study countries & assign population
cntr = C.load("countries")

fuas = (
    pd.concat([fua_jrc, fua_gisco])
    .merge(cntr["icc"], on="icc")
    .to_crs(C.CRS_EU)
)#.view()

#%% Population grid
popu = C.load("popu-grid", quiet=True)
if popu is None:
    C.log("Downloading JRC population grid")
    popu = (
        gpd.read_file("/vsizip//vsicurl/" + C.URLS["popu-grid"],
                      columns=["CNTR_ID", "TOT_P_2018", "geometry"])
        .rename(columns={"CNTR_ID": "icc", "TOT_P_2018": "popu"})
        .astype({"icc": "category", "popu": np.int32})
        .set_index(["icc", "popu"])
        .centroid.rename("geometry")
        .get_coordinates().astype(np.int32)
        .reset_index()
    )#.view()
    C.save(popu, "popu-grid", compression="gzip")
popu = C.pdf2gdf(popu, "x", "y", C.CRS_EU, drop_cols=False)#.view()

#%% Assign population and filter large, populous FUAs
C.log("Filtering populous FUAs")
df = fuas.sjoin(popu[["popu", "x", "y", "geometry"]])
df = df.astype({"x": int, "y": int})
df["x"] *= df["popu"]; df["y"] *= df["popu"]
df = df.groupby("name")[["x", "y", "popu"]].sum()
df["x"] /= df["popu"]; df["y"] /= df["popu"]
df["centre"] = (gpd.points_from_xy(df["x"], df["y"], crs=C.CRS_EU)
                .to_crs(C.CRS_DEG))
fuas2 = (
    fuas.merge(df[["popu", "centre"]], on="name")
    .query(f"popu >= {C.PARAMS['MIN_FUA_POPU']}")
    .reset_index(drop=True)
    .sort_values("name", ignore_index=True)
    .rename_axis("id")
    .reset_index()
    .to_crs(C.CRS_DEG)
    .astype({"id": np.int16, "popu": np.int32})
    [["id", "name", "icc", "popu", "centre", "geometry"]]
)#.view()
C.save(fuas2, "fuas")
