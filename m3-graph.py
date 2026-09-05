"""Combine the modal link tables into the 3MARS transport graph."""

#%% Imports
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
        node_id="AP_" + airports["iata"],
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
    )
    [["src", "trg", "kind", "mode", "operator", "time", "freq"]]
)

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
    [["src", "trg", "kind", "mode", "operator", "time", "freq"]]
)

#%% Bus and rail links between stations
pt = C.load("pt-links")
pt_links = (
    pt.assign(
        src=pt["src"].map(lambda x: f"STN_{x:04}"),
        trg=pt["trg"].map(lambda x: f"STN_{x:04}"),
        kind=pt["intercity"].map({False: "Intraurban", True: "Intercity"}),
        mode=pt["rail"].map({False: "Bus", True: "Rail"}),
    )
    [["src", "trg", "kind", "mode", "operator", "time", "freq"]]
)

#%% Directed car connectors between cities and transport hubs
connectors = C.load("connectors")
connectors["hub_id"] = [
    f"AP_{hub}" if kind == "Airport" else f"STN_{int(hub):04}"
    for hub, kind in zip(connectors["hub"], connectors["kind"])
]
connectors = connectors[connectors["hub_id"].isin(nodes["node_id"])]
connector_links = (
    connectors.assign(
        src=connectors["fua"].map(lambda x: f"FUA_{x:03}"),
        trg=connectors["hub_id"],
        kind="Connector",
        mode="Car",
        operator=pd.NA,
        freq=np.nan,
    )
    [["src", "trg", "kind", "mode", "operator", "time", "freq"]]
)
connector_links = pd.concat([
    connector_links,
    connector_links.rename(columns={"src": "trg", "trg": "src"})
], ignore_index=True)

#%% Combined graph edges
edges = (
    pd.concat(
        [car_links, air_links, pt_links, connector_links],
        ignore_index=True)
    .query("src != trg").reset_index(drop=True)
    .astype({"src": str, "trg": str, "kind": "category",
             "mode": "category", "operator": "string",
             "time": F32, "freq": F32})
)#.view()

#%% Export
C.save(nodes, "3m-nodes")
C.save(edges, "3m-edges")
nodes.to_csv("nodes.csv", index=False)
edges.to_csv("links.csv", index=False)
