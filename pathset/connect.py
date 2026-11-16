"""Car access and egress (e0, en+1) and moves between hubs: two station hubs
of one FUA, or an airport and a station hub of one of its catchment FUAs.

Plan step 2, "Connectors and transfers", first two bullets, and plan,
Definitions, "Hub". Minutes come from the released 3MG graph of 21 Oct
(``flood/nodes.csv``, ``links.csv``: kinds Access FUA -> node, Egress node ->
FUA, Interhub node -> node; all by car). Where the released graph has no
link, minutes = a + b x km, a straight line fitted by least squares on the
released links of the same kind, with km the great-circle distance; such rows
are flagged ``fitted``. Every function is pure; reading and writing is done
by ``pathset-connectors.py``.

For the search (``pathset/search.py``):
- ``access_hubs(connectors, fua)`` -> (hub, acc_min) of an origin FUA and
  ``egress_hubs(connectors, fua)`` -> (hub, egr_min) of a destination FUA;
- ``hub_city(hubs, connectors)`` -> int64 array indexed by hub: the FUA of a
  station hub, the catchment FUA of least access time of an airport, and
  -1 - hub for an airport without a finite access time (its own city). It
  gives the same array as ``search.hub_cities``.
An airport connector may carry NaN in one direction (the direction above
ACCESS_MAX_AIR_MIN); the search drops non-finite minutes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from . import schemas as S
from .network import gc_km, wkb_lonlat

_CONNECTOR_DTYPES = {"fua": "int16", "hub": "int32", "acc_min": "float32",
                     "egr_min": "float32"}
_MOVE_DTYPES = {"hub_a": "int32", "hub_b": "int32", "minutes": "float32"}


# --- Released graph ----------------------------------------------------------
class Released(NamedTuple):
    """Car links of the released graph; minutes are the least over duplicate
    rows of one (src, trg)."""
    access: pd.DataFrame    # fua int, node str, minutes   (FUA -> node)
    egress: pd.DataFrame    # fua int, node str, minutes   (node -> FUA)
    interhub: pd.DataFrame  # src str, trg str, minutes    (node -> node)
    xy: pd.DataFrame        # index node_id; lon, lat


def least_minutes(links: pd.DataFrame, kind: str) -> pd.DataFrame:
    """(src, trg, minutes) of the released car links of one kind; duplicate
    rows of one (src, trg) give their minimum (the release lists an Interhub
    link once per FUA it serves, and some Access links twice). Since the
    graph of 29 October 2026 the kind Interhub also holds links by local
    public transport (mode PT); when ``links`` has a ``mode`` column only its
    car rows are read."""
    l = links[links["kind"] == kind]
    if "mode" in l.columns:
        l = l[l["mode"] == "Car"]
    return (l.groupby(["src", "trg"], as_index=False, sort=True)["time"].min()
            .rename(columns={"time": "minutes"}))


def fua_of_node(node_id: pd.Series) -> np.ndarray:
    """FUA id of a released city node: ``FUA_007`` -> 7."""
    return node_id.str[4:].astype(np.int64).to_numpy()


def released_tables(nodes: pd.DataFrame, links: pd.DataFrame) -> Released:
    """The released Access, Egress and Interhub links and node positions.

    nodes: node_id, lon, lat; links: src, trg, kind, time (minutes) and,
    where the file has it, mode."""
    acc = least_minutes(links, "Access")
    egr = least_minutes(links, "Egress")
    return Released(
        access=pd.DataFrame({"fua": fua_of_node(acc["src"]), "node": acc["trg"],
                             "minutes": acc["minutes"]}),
        egress=pd.DataFrame({"fua": fua_of_node(egr["trg"]), "node": egr["src"],
                             "minutes": egr["minutes"]}),
        interhub=least_minutes(links, "Interhub"),
        xy=nodes.set_index("node_id")[["lon", "lat"]],
    )


def fua_centres(fuas: pd.DataFrame) -> pd.DataFrame:
    """FUA centres (lon, lat) indexed by FUA id; fuas: id, centre (WKB)."""
    lon, lat = wkb_lonlat(fuas["centre"])
    return pd.DataFrame({"lon": lon, "lat": lat},
                        index=fuas["id"].to_numpy(np.int64))


# --- Straight-line fits ------------------------------------------------------
class Fit(NamedTuple):
    """minutes = a + b x km by least squares on n released links."""
    a: float
    b: float
    r2: float
    mae: float       # mean absolute error, minutes
    loo_mae: float   # mean absolute leave-one-out error, minutes
    n: int

    def __call__(self, km) -> np.ndarray:
        return self.a + self.b * np.asarray(km, dtype=np.float64)


def fit_line(km, minutes) -> Fit:
    """Least-squares line through (km, minutes).

    The leave-one-out residual of point i is e_i / (1 - h_i) with the
    leverage h_i = 1/n + (x_i - mean x)^2 / Sxx (exact for ordinary least
    squares; no refitting needed). Raises ValueError with fewer than 3
    points or when all distances are equal (no slope)."""
    x = np.asarray(km, dtype=np.float64)
    y = np.asarray(minutes, dtype=np.float64)
    n = len(x)
    if n < 3:
        raise ValueError(f"a line needs at least 3 points, got {n}")
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        raise ValueError(f"a line needs distinct distances, all {n} are {xm} km")
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    e = y - (a + b * x)
    h = 1.0 / n + (x - xm) ** 2 / sxx
    return Fit(a=float(a), b=float(b),
               r2=float(1.0 - (e ** 2).sum() / ((y - ym) ** 2).sum()),
               mae=float(np.abs(e).mean()),
               loo_mae=float(np.abs(e / (1.0 - h)).mean()), n=n)


def _node_km(xy: pd.DataFrame, a: pd.Series, b: pd.Series) -> np.ndarray:
    pa, pb = xy.reindex(a), xy.reindex(b)
    return gc_km(pa["lon"], pa["lat"], pb["lon"], pb["lat"])


def _fua_node_km(centres: pd.DataFrame, xy: pd.DataFrame, fua, node) -> np.ndarray:
    c, p = centres.reindex(np.asarray(fua)), xy.reindex(node)
    return gc_km(c["lon"], c["lat"], p["lon"], p["lat"])


def released_fits(rel: Released, centres: pd.DataFrame) -> dict[str, Fit]:
    """The five fits of plan step 2, each on the released links of its kind:

    S_acc, S_egr   station Access (FUA -> STN) and Egress (STN -> FUA), km from
                   the FUA centre to the station node; egress has its own fit
                   because the release gives each direction its own minutes;
    A_acc, A_egr   the same for airports (plan: "stations and airports
                   separately");
    move           station-to-station Interhub links, km between the nodes.
    """
    out = {}
    for name, tab, prefix in (("S_acc", rel.access, "STN_"),
                              ("S_egr", rel.egress, "STN_"),
                              ("A_acc", rel.access, "AIR_"),
                              ("A_egr", rel.egress, "AIR_")):
        t = tab[tab["node"].str.startswith(prefix)]
        out[name] = fit_line(_fua_node_km(centres, rel.xy, t["fua"], t["node"]),
                             t["minutes"])
    ih = rel.interhub
    ss = ih[ih["src"].str.startswith("STN_") & ih["trg"].str.startswith("STN_")]
    out["move"] = fit_line(_node_km(rel.xy, ss["src"], ss["trg"]), ss["minutes"])
    return out


# --- Connectors --------------------------------------------------------------
def _with_minutes(pairs: pd.DataFrame, rel: Released, fit_acc: Fit,
                  fit_egr: Fit) -> pd.DataFrame:
    """Attach acc_min and egr_min to pairs (fua, hub, node, km): the released
    minutes of (FUA, node) in each direction where they exist, else the fit
    at km. source = 'released' when both directions are released."""
    key = ["fua", "node"]
    acc = pairs.merge(rel.access, on=key, how="left")["minutes"].to_numpy()
    egr = pairs.merge(rel.egress, on=key, how="left")["minutes"].to_numpy()
    released = ~np.isnan(acc) & ~np.isnan(egr)
    out = pairs[["fua", "hub"]].copy()
    out["acc_min"] = np.where(np.isnan(acc), fit_acc(pairs["km"]), acc)
    out["egr_min"] = np.where(np.isnan(egr), fit_egr(pairs["km"]), egr)
    out["source"] = np.where(released, "released", "fitted")
    return out


def station_connectors(hubs: pd.DataFrame, rel: Released,
                       centres: pd.DataFrame, fits: dict[str, Fit]) -> pd.DataFrame:
    """One connector per station hub, to its own FUA (plan step 2: released
    minutes through the 400 m crosswalk, ps-hubs.released_id; otherwise the
    station fits at the distance FUA centre -> hub)."""
    s = hubs[hubs["kind"] == "S"]
    c = centres.reindex(s["fua"].to_numpy(np.int64))
    pairs = pd.DataFrame({
        "fua": s["fua"].to_numpy(np.int64), "hub": s["hub"].to_numpy(),
        "node": s["released_id"].to_numpy(),
        "km": gc_km(c["lon"], c["lat"], s["lon"], s["lat"])})
    return _with_minutes(pairs, rel, fits["S_acc"], fits["S_egr"])


def airport_connectors(hubs: pd.DataFrame, airports: pd.DataFrame,
                       rel: Released, centres: pd.DataFrame,
                       fits: dict[str, Fit], cap: float) -> pd.DataFrame:
    """Connectors of airports: every FUA of the airport's catchment list
    (data-upstream airports.fua) within ``cap`` minutes (ACCESS_MAX_AIR_MIN;
    plan step 2: "airports serve catchment FUAs within 120 min"). Released
    minutes of FUA_xxx -> AIR_iata and AIR_iata -> FUA_xxx where the release
    has the pair, else the airport fits at the distance FUA centre -> airport.

    The cap holds for each direction: acc_min or egr_min above ``cap`` is set
    to NaN (no access, or no egress, by that airport), and a pair is kept
    when at least one direction is within the cap."""
    a = hubs[hubs["kind"] == "A"][["hub", "iata", "released_id", "lon", "lat"]]
    cat = (airports[["iata", "fua"]].explode("fua").dropna(subset=["fua"])
           .astype({"fua": np.int64}).merge(a, on="iata", how="inner"))
    c = centres.reindex(cat["fua"].to_numpy())
    pairs = pd.DataFrame({
        "fua": cat["fua"].to_numpy(), "hub": cat["hub"].to_numpy(),
        "node": cat["released_id"].to_numpy(),
        "km": gc_km(c["lon"], c["lat"], cat["lon"], cat["lat"])})
    out = _with_minutes(pairs, rel, fits["A_acc"], fits["A_egr"])
    out["acc_min"] = out["acc_min"].where(out["acc_min"] <= cap)
    out["egr_min"] = out["egr_min"].where(out["egr_min"] <= cap)
    return out[out["acc_min"].notna() | out["egr_min"].notna()]


def build_connectors(hubs: pd.DataFrame, airports: pd.DataFrame,
                     rel: Released, centres: pd.DataFrame,
                     fits: dict[str, Fit], cap: float) -> pd.DataFrame:
    """Table ``ps-connectors``: station and airport connectors, one row per
    (fua, hub), sorted by (fua, hub); airport rows may hold NaN in the
    direction above ``cap`` (airport_connectors)."""
    out = pd.concat([station_connectors(hubs, rel, centres, fits),
                     airport_connectors(hubs, airports, rel, centres, fits, cap)],
                    ignore_index=True)
    if out.duplicated(["fua", "hub"]).any():
        raise ValueError("more than one connector per (fua, hub)")
    return (out.sort_values(["fua", "hub"])[S.CONNECTORS]
            .astype(_CONNECTOR_DTYPES).reset_index(drop=True))


def hub_city(hubs: pd.DataFrame, connectors: pd.DataFrame) -> np.ndarray:
    """City of every hub, int64 indexed by hub (plan, Definitions, "Hub": an
    airport's city is the FUA with its shortest access).

    Station hub: ps-hubs.fua. Airport: the FUA of its connector with the
    least finite acc_min, ties to the lower FUA id; without one the airport
    is its own city, -1 - hub (distinct per airport, never a FUA id). Same
    rule and result as ``search.hub_cities``."""
    hub = hubs["hub"].to_numpy(np.int64)
    city = np.empty(len(hubs), dtype=np.int64)
    s = hubs["kind"].to_numpy() == "S"
    city[hub[s]] = hubs["fua"].to_numpy(np.int64)[s]
    city[hub[~s]] = -1 - hub[~s]
    c = connectors[connectors["hub"].isin(hub[~s])
                   & np.isfinite(connectors["acc_min"].to_numpy(np.float64))]
    best = c.sort_values(["hub", "acc_min", "fua"]).drop_duplicates("hub")
    city[best["hub"].to_numpy()] = best["fua"].to_numpy(np.int64)
    return city


def _hubs_with(connectors: pd.DataFrame, fua: int,
               col: str) -> tuple[np.ndarray, np.ndarray]:
    c = connectors[(connectors["fua"] == fua)
                   & np.isfinite(connectors[col].to_numpy(np.float64))]
    c = c.sort_values("hub")
    return c["hub"].to_numpy(np.int32), c[col].to_numpy(np.float32)


def access_hubs(connectors: pd.DataFrame, fua: int) -> tuple[np.ndarray, np.ndarray]:
    """Access hubs of an origin FUA and their acc_min (e0), by hub id; rows
    with NaN acc_min (airport above the cap) are left out."""
    return _hubs_with(connectors, fua, "acc_min")


def egress_hubs(connectors: pd.DataFrame, fua: int) -> tuple[np.ndarray, np.ndarray]:
    """Egress hubs of a destination FUA and their egr_min (en+1), by hub id;
    rows with NaN egr_min (airport above the cap) are left out."""
    return _hubs_with(connectors, fua, "egr_min")


# --- Moves -------------------------------------------------------------------
def station_moves(hubs: pd.DataFrame, rel: Released, fit: Fit,
                  cap: float) -> pd.DataFrame:
    """Directed moves between two station hubs of one FUA (plan step 2:
    "two hubs of one city from the released Interhub link, else the fitted
    model, cap 30 min"). Released minutes when both hubs have a released
    node and the release links them; else the move fit at the distance
    between the hubs. Moves above ``cap`` (MOVE_MAX_MIN) are dropped."""
    s = hubs[hubs["kind"] == "S"][["hub", "fua", "released_id", "lon", "lat"]]
    p = s.merge(s, on="fua", suffixes=("_a", "_b"))
    p = p[p["hub_a"] != p["hub_b"]]
    rel_min = p.merge(rel.interhub, how="left", left_on=["released_id_a",
                      "released_id_b"], right_on=["src", "trg"])["minutes"].to_numpy()
    km = gc_km(p["lon_a"], p["lat_a"], p["lon_b"], p["lat_b"])
    released = ~np.isnan(rel_min)
    out = pd.DataFrame({
        "hub_a": p["hub_a"].to_numpy(), "hub_b": p["hub_b"].to_numpy(),
        "minutes": np.where(released, rel_min, fit(km)),
        "kind": np.where(released, "interhub", "fitted")})
    return out[out["minutes"] <= cap]


def airport_moves(hubs: pd.DataFrame, air2rail: pd.DataFrame, rel: Released,
                  tiers: list[str], cap: float) -> pd.DataFrame:
    """Directed moves between an airport and a station hub (plan step 2).

    colocated  data-upstream-hubs air2rail has (iata, a member station of the
               hub) with a tier in ``tiers`` (AIR2RAIL_COLOCATED_TIERS):
               0 minutes, both directions;
    road       otherwise, the released Interhub link AIR_iata <-> the hub's
               released station node, each direction with its own released
               minutes and kept when at most ``cap`` (AIR_ROAD_MOVE_MAX_MIN).
    No airport-to-airport moves.

    These moves are NOT confined to the airport's city (hub_city): the
    station may lie in any catchment FUA of the airport (the release lists
    airport-station Interhub links for every FUA an airport serves, and a
    co-located station such as Brussels Airport-Zaventem has its own FUA).
    On the 21 Oct release 159 of 567 airport -> station moves (137 road, 22
    colocated) end outside hub_city of the airport, e.g. GRZ -> Maribor.
    The search treats a transfer as the pair of the two hubs' cities.
    Plan step 2 does not restrict airport-station moves to one city; the
    contract line of ps-moves ("of one city") awaits the plan owner."""
    s = hubs[hubs["kind"] == "S"]
    a = hubs[hubs["kind"] == "A"][["hub", "iata", "released_id"]]
    stn2hub = s[["hub", "stns"]].explode("stns").dropna(subset=["stns"])
    stn2hub = pd.Series(stn2hub["hub"].to_numpy(),
                        index=stn2hub["stns"].to_numpy(np.int64))
    col = air2rail[air2rail["tier"].isin(tiers)]
    col = pd.DataFrame({"iata": col["iata"].to_numpy(),
                        "hub_s": stn2hub.reindex(col["stn"].to_numpy(np.int64))
                        .to_numpy()}).dropna()
    col = (col.merge(a, on="iata")[["hub", "hub_s"]].astype(np.int64)
           .drop_duplicates())
    co = pd.concat([
        pd.DataFrame({"hub_a": col["hub"], "hub_b": col["hub_s"]}),
        pd.DataFrame({"hub_a": col["hub_s"], "hub_b": col["hub"]})])
    co = co.assign(minutes=0.0, kind="colocated")

    sr = s.loc[s["released_id"] != "", ["hub", "released_id"]]
    ar = a.loc[a["released_id"] != "", ["hub", "released_id"]]
    ih = rel.interhub
    a2s = (ih.merge(ar, left_on="src", right_on="released_id")
           .merge(sr, left_on="trg", right_on="released_id", suffixes=("_a", "_b")))
    s2a = (ih.merge(sr, left_on="src", right_on="released_id")
           .merge(ar, left_on="trg", right_on="released_id", suffixes=("_a", "_b")))
    road = pd.concat([a2s, s2a])
    road = pd.DataFrame({"hub_a": road["hub_a"].to_numpy(),
                         "hub_b": road["hub_b"].to_numpy(),
                         "minutes": road["minutes"].to_numpy(), "kind": "road"})
    road = road[road["minutes"] <= cap]
    pair = pd.MultiIndex.from_frame(road[["hub_a", "hub_b"]])
    road = road[~pair.isin(pd.MultiIndex.from_frame(co[["hub_a", "hub_b"]]))]
    return pd.concat([co, road], ignore_index=True)


def build_moves(hubs: pd.DataFrame, air2rail: pd.DataFrame, rel: Released,
                fits: dict[str, Fit], params: dict) -> pd.DataFrame:
    """Table ``ps-moves``: station-station and airport-station moves, one row
    per directed hub pair, sorted by (hub_a, hub_b)."""
    out = pd.concat([
        station_moves(hubs, rel, fits["move"], params["MOVE_MAX_MIN"]),
        airport_moves(hubs, air2rail, rel, params["AIR2RAIL_COLOCATED_TIERS"],
                      params["AIR_ROAD_MOVE_MAX_MIN"])], ignore_index=True)
    if out.duplicated(["hub_a", "hub_b"]).any():
        raise ValueError("more than one move per (hub_a, hub_b)")
    return (out.sort_values(["hub_a", "hub_b"])[S.MOVES]
            .astype(_MOVE_DTYPES).reset_index(drop=True))
