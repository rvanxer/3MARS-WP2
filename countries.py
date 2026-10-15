#%% Setup
import pandas as pd
import geopandas as gpd

import config as C

#%% Countries in the study region
# given by 2-letter International Country Code (ICC), name & largest city
imp_countries = [
    ("AT", "Austria", "Vienna"),
    ("BE", "Belgium", "Brussels"),
    ("BG", "Bulgaria", "Sofia"),
    ("CH", "Switzerland", "Zurich"),
    ("CZ", "Czechia", "Prague"),
    ("DE", "Germany", "Berlin"),
    ("DK", "Denmark", "Copenhagen"),
    ("EE", "Estonia", "Tallinn"),
    ("EL", "Greece", "Athens"),
    ("ES", "Spain", "Madrid"),
    ("FI", "Finland", "Helsinki"),
    ("FR", "France", "Paris"),
    ("HR", "Croatia", "Zagreb"),
    ("HU", "Hungary", "Budapest"),
    ("IE", "Ireland", "Dublin"),
    ("IT", "Italy", "Rome"),
    ("LT", "Lithuania", "Vilnius"),
    ("LU", "Luxembourg", "Luxembourg"),
    ("LV", "Latvia", "Riga"),
    ("NL", "Netherlands", "Amsterdam"),
    ("NO", "Norway", "Oslo"),
    ("PL", "Poland", "Warsaw"),
    ("PT", "Portugal", "Lisbon"),
    ("RO", "Romania", "Bucharest"),
    ("SE", "Sweden", "Stockholm"),
    ("SI", "Slovenia", "Ljubljana"),
    ("SK", "Slovakia", "Bratislava"),
    ("UK", "United Kingdom", "London"),
]

#%% Country boundaries: Europe
"""For all study countries except the UK, the country boundaries are 
downloaded from the 2024 "Territorial Units for Statistics" (NUTS) level 0 & 1 
boundaries from the data portal of the Geographic Information System of the 
EU Commission (GISCO) with spatial precision of 10 m."""
C.log("Downloading NUTS level 0 & 1 boundaries")
nuts0 = (
    gpd.read_file(C.URLS["regions-nuts0"])
    .rename(columns={
        "CNTR_CODE": "icc",
        "NUTS_ID": "geoid",
        "NAME_LATN": "name",
    }).clip((-10, 30, 50, 75))
    [["geoid", "icc", "geometry"]]
)
nuts0["name"] = nuts0["icc"].map({x[0]: x[1] for x in imp_countries})
nuts0 = nuts0.dropna(subset="name")

#%% Country boundaries: UK [0:10]
"""For the special case of the UK, the boundaries are downloaded from the 
International Territorial Level (ITL) data layer from the  UK's Office for 
National Statistics website at the Generalised Clipped Boundary Level 1."""
C.log("Downloading UK ITL-1 boundaries")
itl = gpd.read_file(C.URLS["regions-itl1"])
itl = itl.rename(columns={"ITL125CD": "geoid"})
itl.geometry = itl.simplify(0.1) # round off geometry to 0.1°
itl0 = itl.dissolve().assign(icc="UK", name="United Kingdom")

#%% Countries
cntr = pd.concat([nuts0, itl0])
cntr["timezone"] = cntr["icc"].map({
    x[0]: "Europe/" + x[2] for x in imp_countries
})
cntr = cntr.sort_values("icc", ignore_index=True)
cntr = cntr[["icc", "name", "timezone", "geometry"]]
C.save(cntr, "countries")
