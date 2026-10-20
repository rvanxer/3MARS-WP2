"""Combine the modal link tables into the 3MARS transport graph."""

#%% Imports
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

F32 = np.float32

#%% Load nodes
fuas = C.load("fuas")
airports = C.load("airports")
stns = C.load("ic-stations")
lines = C.load("ic-lines", columns=["rail", "stn"])

stn_kind = (
    lines.explode("stn").astype({"stn": int})
    .groupby("stn")["rail"].agg(lambda x: tuple(sorted(set(x))))
    .map({
        (False,): "Bus station",
        (True,): "Rail station",
        (False, True): "Multimodal station",
    })
)

#%% Graph nodes
city_nodes = (
    fuas.assign(
        node_id=fuas["id"].map(lambda x: f"FUA_{x:03}"),
        kind="City",
        local_id=fuas["id"].astype(str),
        name=fuas["name"],
        lon=fuas["centre"].x,
        lat=fuas["centre"].y,
    )
    [["node_id", "kind", "local_id", "name", "lon", "lat"]]
)
airport_nodes = (
    airports.assign(
        node_id="AIR_" + airports["iata"],
        kind="Airport",
        local_id=airports["iata"],
        name=airports["name"],
        lon=airports.geometry.x,
        lat=airports.geometry.y,
    )
    [["node_id", "kind", "local_id", "name", "lon", "lat"]]
)
station_nodes = (
    stns.assign(
        node_id=stns["stn"].map(lambda x: f"STN_{x:04}"),
        kind=stns["stn"].map(stn_kind),
        local_id=stns["stn"].astype(str),
        name=stns["name"].str.split(" | ", n=1, regex=False).str[0],
        lon=stns.geometry.x,
        lat=stns.geometry.y,
    )
    [["node_id", "kind", "local_id", "name", "lon", "lat"]]
)
nodes = (
    pd.concat([city_nodes, airport_nodes, station_nodes], ignore_index=True)
    .astype({"node_id": str, "kind": "category", "local_id": str,
             "name": str, "lon": F32, "lat": F32})
)#.view()

#%% Car links between cities
car = C.load("car-links")
car_links = (
    car.assign(
        src=car["src_fua"].map(lambda x: f"FUA_{x:03}"),
        trg=car["trg_fua"].map(lambda x: f"FUA_{x:03}"),
        kind="Intercity",
        mode="Car",
        operator=pd.NA,
        freq=np.nan,
    ).astype({"dist": np.int16})
    [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

#%% Air links between airports
air = C.load("air-links")
air_links = (
    air.assign(
        src="AIR_" + air["src"].astype(str),
        trg="AIR_" + air["trg"].astype(str),
        kind="Intercity",
        mode="Air",
        operator=air["carrier"].astype(str),
    )
    [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

#%% Bus and rail links between stations
pt = C.load("pt-links")
pt_links = (
    pt.assign(
        src=pt["src"].map(lambda x: f"STN_{x:04}"),
        trg=pt["trg"].map(lambda x: f"STN_{x:04}"),
        mode=pt["rail"].map({False: "Bus", True: "Rail"}),
        kind="Intercity",
    ).query("intercity")
    [["src", "trg", "kind", "mode", "operator", "dist", "time", "freq"]]
)#.view()

#%% Directed car connectors within cities
connectors = (
    C.load("connectors")
    .assign(mode="Car", operator=pd.NA, freq=np.nan)
    .astype({"dist": np.int32})
)#.view()

#%% Combined graph edges
edges = (
    pd.concat([car_links, air_links, pt_links, connectors], ignore_index=True)
    .query("src != trg").reset_index(drop=True)
    .astype({"src": str, "trg": str, "kind": "category",
             "mode": "category", "dist": F32, "time": F32, "freq": F32})
)
edges["operator"] = edges["operator"].fillna("<NA>").astype("category")

#%% Merge nearby airports & stations
hubs = gpd.sjoin(
    (nodes[nodes["kind"].str.contains("station")]
     .pipe(C.pdf2gdf, crs=C.CRS_DEG).to_crs(C.CRS_EU)
     [["node_id", "geometry"]]
     .rename(columns={"node_id": "station"})),
    (nodes[nodes["kind"] == "Airport"]
     .pipe(C.pdf2gdf, crs=C.CRS_DEG).to_crs(C.CRS_EU)
     [["node_id", "name", "geometry"]]),
    predicate="dwithin",
    distance=C.PARAMS["AIRPT_STN_DIST"],
)
nodes2 = nodes[~nodes["node_id"].isin(hubs["station"])].reset_index(drop=True)
stn2hub = dict(zip(hubs["station"], hubs["node_id"]))
edges2 = edges.copy()
for col in ["src", "trg"]:
    idx = edges2[edges2[col].isin(hubs["station"])].index
    edges2.loc[idx, col] = edges2.loc[idx, col].map(stn2hub)
edges2 = edges2.query("src != trg").reset_index(drop=True)#.view()

#%% Export
C.save(nodes2, "3m-nodes")
C.save(edges2, "3m-edges")
nodes2.to_csv("nodes.csv", index=False)
edges2.to_csv("links.csv", index=False)
