"""Approximate geometry of consecutive station segments by routing along modal
OSM network."""

#%%  Imports
from collections import defaultdict
import heapq
import warnings

import igraph as ig
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import MultiLineString, line_merge, get_parts
from tqdm import tqdm

import config as C

params = C.load_params()

#%% Create base modal graph
def create_graph(mode: str, islands={
    "Ireland": (-11.563513, 51.283195, -5.257361, 55.535445),
    "Sardinia": (7.579311, 38.780768, 10.093450, 41.342679),
}.values()) -> ig.Graph:
    ## Load OSM network for the mode
    fname = {"Bus": "highways", "Rail": "railways"}[mode]
    ways = C.load(f"osm/{fname}").reset_index(drop=True).rename_axis("way")
    xy = ways.get_coordinates()
    grps = xy.groupby("way", sort=False)
    endpts = pd.concat([grps.agg("first"), grps.agg("last")])
    nodes = (
        endpts.drop_duplicates(ignore_index=True)
        .rename_axis("junc")
        .reset_index()
        .assign(island=0)
    )
    for i, (x0, y0, x1, y1) in enumerate(islands):
        df = nodes.query(f"{x0} <= x <= {x1} & {y0} <= y <= {y1}")
        nodes.loc[df.index, "island"] = i + 1
    junc2way = (
        endpts.reset_index()
        .merge(nodes, on=["x", "y"])
        .sort_values("way", ignore_index=True)
        [["junc", "way"]]
    )
    src = junc2way.drop_duplicates("way", keep="first")
    src = src.rename(columns={"junc": "source"})
    trg = junc2way.drop_duplicates("way", keep="last")
    trg = trg.rename(columns={"junc": "target"})
    edges = (
        src.merge(trg, on="way")
        .merge(ways, on="way")
        [["source", "target", "way", "len_km", "geometry"]]
    )
    g = ig.Graph.DataFrame(edges, directed=False, vertices=nodes)
    nodes = g.get_vertex_dataframe()
    imp_junc_ids = []
    for _, df in nodes.groupby("island"):
        lcc = g.subgraph(df.index).components().giant()
        imp_junc_ids.extend(lcc.vs["junc"])
    imp_nodes = nodes[nodes["junc"].isin(imp_junc_ids)].index
    g = g.subgraph(imp_nodes)
    C.log(f"Base {mode} graph: |V|={g.vcount():,}, |E|={g.ecount():,}")
    return g

g0_b = create_graph("Bus") # 10s
g0_r = create_graph("Rail") # 6s

#%% Contract edges
# Simplify the graph by combining all 2-degree node chains
def contract_edges(g: ig.Graph) -> ig.Graph:
    C.log("Contracting 2-degree nodes")
    important = [d != 2 for d in g.degree()]
    visited = [False] * g.ecount()
    new_edges, ways, length, geometries = [], [], [], []
    
    def add_edges(nodes, edges):
        new_edges.append((nodes[0], nodes[-1]))
        ways.append([g.es[eid]["way"] for eid in edges])
        length.append(sum([g.es[eid]["len_km"] for eid in edges]))
        geoms = [g.es[eid]["geometry"] for eid in edges]
        for i in range(len(geoms)):
            if geoms[i].geom_type == "MultiLineString":
                geoms[i] = line_merge(geoms[i])
                if geoms[i].geom_type == "MultiLineString":
                    geoms[i] = get_parts(geoms[i])[0]
        geometries.append(MultiLineString(geoms))
    
    for v in range(g.vcount()):
        if not important[v]:
            continue
        for eid in g.incident(v):
            if visited[eid]:
                continue
            path_edges = []
            path_vertices = [v]
            current_v = v
            while True:
                visited[eid] = True
                path_edges.append(eid)
                e = g.es[eid]
                next_v = e.target if e.source == current_v else e.source
                path_vertices.append(next_v)
                if important[next_v]:
                    break
                nbr_edges = g.incident(next_v)
                next_eid = nbr_edges[0] if nbr_edges[1] == eid else nbr_edges[1]
                current_v = next_v
                eid = next_eid
            add_edges(path_vertices, path_edges)
    # handle pure cycles (all degree-2 components)
    for eid in range(g.ecount()):
        if visited[eid]:
            continue
        path_edges = [eid]
        visited[eid] = True
        e = g.es[eid]
        start = e.source
        current_v = e.target
        path_vertices = [start, current_v]
        while current_v != start:
            nbr_edges = g.incident(current_v)
            next_eid = (nbr_edges[0] if nbr_edges[1] == path_edges[-1] 
                        else nbr_edges[1])
            if visited[next_eid]:
                break
            visited[next_eid] = True
            path_edges.append(next_eid)
            e = g.es[next_eid]
            next_v = e.target if e.source == current_v else e.source
            path_vertices.append(next_v)
            _, current_v = current_v, next_v
        add_edges(path_vertices, path_edges)
    # build new graph
    g2 = ig.Graph()
    g2.add_vertices(g.vcount())
    for attr in g.vs.attributes():
        g2.vs[attr] = g.vs[attr]
    g2.add_edges(new_edges)
    g2.es["way"] = ways
    g2.es["len_km"] = length
    g2.es["geometry"] = geometries
    # remove unused vertices
    used = set(v for e in new_edges for v in e)
    g = g2.subgraph(list(used))
    C.log(f"Contracted graph: |V|={g.vcount():,}, |E|={g.ecount():,}")
    return g

g_b = contract_edges(g0_b) # 19s
g_r = contract_edges(g0_r) # 7s

#%% Stations snapped to nodes
def get_stations(
    graph: ig.Graph,
    max_snap_dist: float = params.MAX_STN_OSM_OFFSET
) -> pd.DataFrame:
    C.log("Snapping stations to graph nodes")
    stns = (
        C.load("ic-stations")
        .to_crs(C.CRS_EU)
        .set_index(["stn", "name"])
        .get_coordinates()
        .reset_index()
        .set_index("stn")
        .sort_index()
    )
    nodes = (
        graph.get_vertex_dataframe()
        .rename_axis("node")
        .reset_index()
        .pipe(C.pdf2gdf, "x", "y", C.CRS_DEG)
        .to_crs(C.CRS_EU)
        .set_index(["node", "junc", "island"])
        .get_coordinates()
        .reset_index(["junc", "island"])
    )
    stn2junc = []
    for _, nodes_ in nodes.groupby("island"):
        nodes_ = (
            graph.subgraph(nodes_.index)
            .components().giant()
            .get_vertex_dataframe()
            .rename_axis("node").reset_index()
            [["node", "junc"]]
            .merge(nodes[["junc", "x", "y"]], on="junc")
        )
        tree = cKDTree(nodes_[["x", "y"]])
        dist, index = tree.query(
            stns[["x", "y"]], k=1,
            distance_upper_bound=max_snap_dist
        )
        df = pd.DataFrame({
            "stn": stns.index,
            "node": np.hstack([nodes_.index, [-1]])[index],
            "snap_dist": dist
        })
        df = df[df["node"] != -1]
        df = df.merge(nodes_[["node", "junc"]], on="node")
        stn2junc.append(df[["stn", "junc", "snap_dist"]])
    return (
        pd.concat(stn2junc)
        .sort_values("snap_dist")
        .groupby("stn")
        .head(1)
        .merge(nodes.reset_index(), on="junc")
        [["stn", "node"]]
        .merge(stns, on="stn")
        .sort_values("stn", ignore_index=True)
    )

stns_b = get_stations(g_b) # 3s
stns_r = get_stations(g_r) # 1s

#%% Greedily orient pairs
def orient_pairs_greedily(pairs: pd.DataFrame) -> pd.DataFrame:
    """Orient undirected (u, v) pairs to reduce the number of Dijkstra sources.
    This is a greedy vertex-cover heuristic. It repeatedly selects the node
    incident to the largest number of still-unassigned pairs.
    """
    pairs = pairs.reset_index(drop=True).copy()
    uv = pairs[["u", "v"]].to_numpy(dtype=np.int32)
    n = len(uv)
    incident = defaultdict(list)
    for pair_id, (u, v) in enumerate(uv):
        incident[int(u)].append(pair_id)
        incident[int(v)].append(pair_id)
    active = np.ones(n, dtype=bool)
    degree = {node: len(ids) for node, ids in incident.items()}
    heap = [(-value, node) for node, value in degree.items()]
    heapq.heapify(heap)
    route_src = np.full(n, -1, dtype=np.int32)
    route_trg = np.full(n, -1, dtype=np.int32)
    while heap:
        negative_degree, source = heapq.heappop(heap)
        current_degree = -negative_degree
        if degree[source] != current_degree:
            continue # ignore obsolete heap entries
        if current_degree == 0:
            break
        for pair_id in incident[source]:
            if not active[pair_id]:
                continue
            active[pair_id] = False
            u, v = uv[pair_id]
            route_src[pair_id] = source
            route_trg[pair_id] = v if u == source else u
            for endpoint in (int(u), int(v)):
                degree[endpoint] -= 1
                heapq.heappush(heap, (-degree[endpoint], endpoint))
    if active.any():
        raise RuntimeError("Some OD pairs were not oriented")
    pairs["route_src"], pairs["route_trg"] = route_src, route_trg
    return pairs

#%% Segment geometry
def get_shortest_paths(
    mode: str,
    graph: ig.Graph,
    stns: pd.DataFrame,
    round_tol=100 # metres
) -> pd.DataFrame:
    C.log("Computing shortest paths")
    edges = graph.get_edge_dataframe().rename_axis("edge_id")
    edges = gpd.GeoDataFrame(edges, crs=C.CRS_DEG).to_crs(C.CRS_EU)
    seg = C.load("ic-segments")
    seg = seg[seg.pop("mode").isin([mode, "Both"])]
    seg["source"] = seg["src"].map(stns.set_index("stn")["node"])
    seg["target"] = seg["trg"].map(stns.set_index("stn")["node"])
    seg = seg.dropna(subset=["source", "target"])
    seg = seg.astype({"source": int, "target": int})
    comp = np.array(graph.connected_components().membership)
    seg["u"] = np.minimum(seg["source"], seg["target"])
    seg["v"] = np.maximum(seg["source"], seg["target"])
    pairs = seg.loc[(seg["u"] != seg["v"]) & 
                    (comp[seg["source"]] == comp[seg["target"]])]
    pairs = pairs[["u", "v"]].drop_duplicates()
    pairs = orient_pairs_greedily(pairs)
    od = [] # unique undirected OD pairs
    for src, df in tqdm(pairs.groupby("route_src", sort=False)):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Couldn't reach some vertices.*")
            edge_ids = graph.get_shortest_paths(
                v=src, to=list(df["route_trg"]),
                weights="len_km", output="epath"
            )
        for u, v, eids in zip(df["u"], df["v"], edge_ids):
            if len(eids) > 0:
                length = edges.loc[eids]["len_km"].sum()
                g = line_merge(edges.loc[eids].union_all()).simplify(round_tol)
                od.append({"u": u, "v": v, "len_km": length, "geometry": g})
    od = gpd.GeoDataFrame(od, crs=C.CRS_EU).to_crs(C.CRS_DEG)
    od = od.merge(seg, "right", on=["u", "v"]).assign(mode=mode)
    return od[["src", "trg", "mode", "len_km", "line", "geometry"]]

seg_b = get_shortest_paths("Bus", g_b, stns_b)#.view() # 41s
seg_r = get_shortest_paths("Rail", g_r, stns_r)#.view() # 9s

#%% Combine segments and export
seg = (pd.concat([seg_b, seg_r], ignore_index=True)
       .rename_axis("seg").reset_index())#.view()
C.save(seg, "ic-segments")
