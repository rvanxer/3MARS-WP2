"""Combine the modal link tables into the 3MARS transport graph."""

#%% Imports
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

F32 = np.float32

#%% Nodes: Cities & airports
fuas = (
    C.load("fuas").set_index("id")
    [["node_id", "name", "centre"]]
    .assign(lat=lambda df: df["centre"].y,
            lon=lambda df: df["centre"].x)
    .drop(columns="centre")
    .assign(kind="City")
)#.view()
airports = (
    C.load("airports")
    .set_index(["iata", "node_id", "name"])
    .get_coordinates()
    .set_axis(["lon", "lat"], axis=1)
    .reset_index()
    .assign(kind="Airport")
    .set_index("iata").rename_axis("id")
)#.view()

#%% Nodes: Stations
stations = (
    C.load("ic-stations")
    .set_index(["stn", "node_id", "name"])
    .get_coordinates()
    .set_axis(["lon", "lat"], axis=1)
    .reset_index()
)
stations["name"] = stations["name"].str.split(" | ", regex=False).str[0]
stations.insert(2, "kind", stations["stn"].map(
    C.load("ic-lines", columns=["rail", "stn"])
    .explode("stn").astype({"stn": int, "rail": int})
    .groupby("stn")
    ["rail"].agg(lambda x: tuple(sorted(set(x))))
    .map({(0,): "Bus", (1,): "Rail", (0, 1): "Multimodal"})
) + " station")
stations = stations.set_index("stn").rename_axis("id")#.view()

#%% Map stations close to airports to their multimodal airports
stn2ap = gpd.sjoin(
    (C.pdf2gdf(stations, crs=C.CRS_DEG).to_crs(C.CRS_EU)
    .rename(columns={"node_id": "station"})),
    (C.pdf2gdf(airports, crs=C.CRS_DEG).to_crs(C.CRS_EU)
    .rename(columns={"node_id": "airport"})),
    predicate="dwithin",
    distance=C.PARAMS["AIRPT_STN_DIST"]
).set_index("station")["airport"].to_dict()
C.log(f"{len(stn2ap)} stations mapped to nearby airports")

#%% Combine nodes
nodes = (
    pd.concat([fuas, airports, stations])
    .set_index("node_id")
    .drop(index=list(stn2ap))
    [["kind", "name", "lat", "lon"]]
    .astype({"name": str, "lat": F32, "lon": F32})
    .reset_index()
)#.view()

#%% Service links: Intercity air, car & public transport (PT)
ic_car_links = (
    C.load("car-links")
    .assign(src=lambda df: df["src_fua"].map(fuas["node_id"]),
            trg=lambda df: df["trg_fua"].map(fuas["node_id"]),
            kind="Intercity",
            mode="Car",
            operator=pd.NA,
            freq=np.nan)
    # [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

air_links = (
    C.load("air-links")
    .assign(kind="Intercity", mode="Air")
    .rename(columns={"avg_time": "time", "avg_freq": "freq"})
    # [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

ic_pt_links = (
    C.load("pt-links")
    .query("intercity")
    .assign(src=lambda df: df["src"].map(stations["node_id"]),
            trg=lambda df: df["trg"].map(stations["node_id"]),
            mode=lambda df: df["rail"].map({True: "Rail", False: "Bus"}),
            kind="Intercity")
    # [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

#%% Local links: Connectors & interhub car & PT links
local_car_links = (
    C.load("connectors")
    .assign(mode="Car", operator=None, freq=np.nan)
)#.view()

local_pt_links = (
    C.load("local-pt-links")
    .assign(kind="Interhub", mode="PT", operator=None, dist=np.nan)
    .rename(columns={"avg_time": "time", "avg_freq": "freq"})
)#.view()

#%% Combine links and map back stations to nearby airports
links = (
    pd.concat([
        air_links,
        ic_pt_links,
        ic_car_links,
        local_car_links,
        local_pt_links
    ])[["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
    .astype({"dist": F32, "time": F32, "freq": F32})
    .assign(src=lambda df: df["src"].replace(stn2ap))
    .assign(trg=lambda df: df["trg"].replace(stn2ap))
    .query("src != trg")
    .reset_index(drop=True)
)#.view()

#%% Export
C.save(nodes, "3m-nodes")
C.save(links, "3m-edges")
nodes.to_csv("nodes.csv", index=False)
links.to_csv("links.csv", index=False)

#%%
