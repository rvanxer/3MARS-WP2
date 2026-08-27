#%% Setup
import pandas as pd
import geopandas as gpd

import config as C

#%% Countries in the study region
# given by 2-letter International Country Code (ICC) and name
imp_countries = dict(
    AT = "Austria",
    BE = "Belgium",
    BG = "Bulgaria",
    CH = "Switzerland",
    CZ = "Czechia",
    DE = "Germany",
    DK = "Denmark",
    EE = "Estonia",
    EL = "Greece",
    ES = "Spain",
    FI = "Finland",
    FR = "France",
    HR = "Croatia",
    HU = "Hungary",
    IE = "Ireland",
    IT = "Italy",
    LT = "Lithuania",
    LU = "Luxembourg",
    LV = "Latvia",
    NL = "Netherlands",
    NO = "Norway",
    PL = "Poland",
    PT = "Portugal",
    RO = "Romania",
    SE = "Sweden",
    SI = "Slovenia",
    SK = "Slovakia",
    UK = "United Kingdom",
)

#%% Country boundaries: Europe
"""For all study countries except the UK, the country boundaries are 
downloaded from the 2024 "Territorial Units for Statistics" (NUTS) level 0 & 1 
boundaries from the data portal of the Geographic Information System of the 
EU Commission (GISCO) with spatial precision of 10 m."""
C.log("Downloading NUTS level 0 & 1 boundaries")
nuts0 = (
    gpd.read_file("https://gisco-services.ec.europa.eu/distribution/v2/"
                    f"nuts/gpkg/NUTS_RG_10M_2024_4326_LEVL_0.gpkg")
    .rename(columns={
        "CNTR_CODE": "icc",
        "NUTS_ID": "geoid",
        "NAME_LATN": "name",
    }).clip((-10, 30, 50, 75))
    [["geoid", "icc", "geometry"]]
)
nuts0["name"] = nuts0["icc"].map(imp_countries)
nuts0 = nuts0.dropna(subset="name")

#%% Country boundaries: UK [0:10]
"""For the special case of the UK, the boundaries are downloaded from the 
International Territorial Level (ITL) data layer from the  UK's Office for 
National Statistics website at the Generalised Clipped Boundary Level 1."""
C.log("Downloading UK ITL-1 boundaries")
itl = gpd.read_file(
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/"
    "rest/services/ITL1_JAN_2025_UK_BGC/FeatureServer/0/query?"
    "where=1%3D1&returnGeometry=true&f=geojson"
).rename(columns={"ITL125CD": "geoid"})
itl.geometry = itl.simplify(0.1) # round off geometry to 0.1°

itl0 = itl.dissolve().assign(icc="UK", name="United Kingdom")

#%% Countries
countries = (pd.concat([nuts0, itl0])
             .sort_values("icc", ignore_index=True)
             [["icc", "name", "geometry"]])
C.save(countries, "countries")
