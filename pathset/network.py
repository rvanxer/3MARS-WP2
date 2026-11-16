"""Static service network: hubs, trips, legs, runs, riding-time classes, links.

Plan step 1 and its Definitions (Hub, Trip, Near-duplicate, Service link,
Active date, Link frequency). Every function here is a pure function of
DataFrames and arrays; reading and writing is done by ``pathset-network.py``.

Conventions inside this module
- A mode is an ``int8`` code, the index in ``schemas.MODES`` (A = 0, B = 1,
  R = 2); it becomes a letter only in the written tables.
- A calendar is a row of ``dates.N_WORDS`` uint64 words (``pathset.dates``).
- "Local" minutes are wall-clock minutes at the hub concerned; "elapsed"
  minutes are local minutes less 60 x the hub's standard offset, a clock on
  which differences are riding times.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo
import datetime as dt

import numpy as np
import pandas as pd
import shapely
from numba import njit

from . import dates as D
from . import schemas as S
from .dates import and_any, and_count, or_into, popcount_row, shift_row

MODE_A, MODE_B, MODE_R = (np.int8(S.MODES.index(m)) for m in "ABR")
MODE_LETTERS = np.array(S.MODES)

EARTH_R_KM = 6371.0088

# Standard-time UTC offset of each study country, hours. One zone per country
# is exact here: every FUA and airport lies on the European mainland or the
# British Isles. Checked against the time zones of the 3M graph pipeline by
# ``pipeline_country_tz``.
TZ_STD = {
    "UK": 0, "IE": 0, "PT": 0,
    "AT": 1, "BE": 1, "CH": 1, "CZ": 1, "DE": 1, "DK": 1, "ES": 1, "FR": 1,
    "HR": 1, "HU": 1, "IT": 1, "LU": 1, "NL": 1, "NO": 1, "PL": 1, "SE": 1,
    "SI": 1, "SK": 1,
    "BG": 2, "EE": 2, "EL": 2, "FI": 2, "LT": 2, "LV": 2, "RO": 2,
}

# GTFS route types counted as rail and as bus; copied from intercity.py
# lines 36-44 (cell "Routes").
RAIL_ROUTE_TYPES = (2, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111,
                    113, 114, 117, 900, 901, 902, 903, 904, 905, 906)
BUS_ROUTE_TYPES = (3, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                   700, 701, 702, 703, 704, 705, 706, 707, 708,
                   709, 710, 711, 712, 713, 714, 715, 716)

# Fixed road connections across water; copied from car-times.py lines 33-39:
# (city A, city B, km A->B, minutes A->B, km B->A, minutes B->A).
CAR_CROSSINGS = (
    ("Medway", "Dunkerque", 178, 164, 183, 161),          # English Channel
    ("Messina", "Reggio di Calabria", 24, 58, 24, 49),    # Strait of Messina
    ("Helsinki", "Tallinn", 180, 87.7, 182, 88.3),        # Gulf of Finland
)

# Reasons of the hygiene screen (plan step 1, "Drops"); code 0 = kept.
HYGIENE_REASONS = ("kept", "ride <= 0", "ride > MAX_RIDE_MIN",
                   "speed above V_MAX_KMH")

_DTYPES = {
    "hubs": {"hub": "int32", "fua": "int16", "tz_std": "int8",
             "lon": "float64", "lat": "float64", "match_m": "float32"},
    "links": {"link": "int32", "src": "int32", "trg": "int32", "cls": "int16",
              "ride": "float32", "ride_min": "int16", "ride_max": "int16",
              "tz_shift": "int16", "km": "float32", "ev0": "int32",
              "ev1": "int32", "n_events": "int32", "active_dates": "int32",
              "first_day": "int32", "last_day": "int32", "freq": "float32",
              "freq_max": "int16", "freq_span": "float32",
              "freq_fixed": "float32", "active_share": "float32"},
    "events": {"link": "int32", "dep": "int16", "ride": "int16",
               "trip": "int32", "n_dates": "int32"},
    "members": {"link": "int32", "route_type": "int16", "line": "int64",
                "n_events": "int32"},
    "drops": {"n": "int64"},
    "car": {"o": "int16", "d": "int16", "minutes": "float32", "km": "float32",
            "fixed": "bool"},
}


def as_contract(df: pd.DataFrame, table: str, columns: list[str]) -> pd.DataFrame:
    """Columns in the order of ``schemas`` with the dtypes of the contract."""
    return df[columns].astype(
        {c: t for c, t in _DTYPES[table].items() if c in columns}
    ).reset_index(drop=True)


# --- Geometry ----------------------------------------------------------------
def wkb_lonlat(wkb) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of WKB points (EPSG:4326)."""
    pts = shapely.from_wkb(np.asarray(wkb, dtype=object))
    return shapely.get_x(pts), shapely.get_y(pts)


def gc_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km (haversine, mean Earth radius)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --- a. Hubs -----------------------------------------------------------------
def single_linkage(group, lon, lat, radius_m: float) -> np.ndarray:
    """Cluster points by single linkage inside each group.

    Two points get the same label when they belong to the same group and a
    chain of points of that group joins them with every step at most
    ``radius_m`` (great-circle). Plan, Definitions, "Hub": stations of one FUA
    linked within 400 m. Labels are numbered by first appearance.
    """
    group, lon, lat = np.asarray(group), np.asarray(lon), np.asarray(lat)
    label = np.full(len(group), -1, dtype=np.int64)
    nxt = 0
    for g in pd.unique(group):
        idx = np.flatnonzero(group == g)
        near = gc_km(lon[idx][:, None], lat[idx][:, None],
                     lon[idx][None, :], lat[idx][None, :]) * 1000 <= radius_m
        for a in range(len(idx)):
            if label[idx[a]] >= 0:
                continue
            label[idx[a]] = nxt
            stack = [a]
            while stack:
                u = stack.pop()
                for v in np.flatnonzero(near[u]):
                    if label[idx[v]] < 0:
                        label[idx[v]] = nxt
                        stack.append(v)
            nxt += 1
    return label


def _assigned(script: Path, name: str):
    """The expression assigned to ``name`` at the top level of a pipeline
    script, or None. The script is parsed, not run (the pipeline scripts run
    on import)."""
    if not script.exists():
        return None
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == name):
            return node.value
    return None


def pipeline_country_tz(root: Path) -> dict:
    """Winter UTC offset (hours) per country as the 3M graph pipeline in
    ``root`` defines it, read from the source text of its scripts.

    Since October 2026 countries.py lists (country code, name, largest city)
    in ``imp_countries`` and the zone of a country is "Europe/" + that city.
    The amended air-times.py of September 2026 holds a ``COUNTRY_TZ`` table
    instead; it is read when countries.py names no cities. Offsets are taken
    on 15 January 2025."""
    root = Path(root)
    zones = None
    value = _assigned(root / "countries.py", "imp_countries")
    if isinstance(value, (ast.List, ast.Tuple)):
        rows = ast.literal_eval(value)
        if rows and all(isinstance(r, tuple) and len(r) >= 3 for r in rows):
            zones = {r[0]: "Europe/" + r[2] for r in rows}
    if zones is None:
        value = _assigned(root / "air-times.py", "COUNTRY_TZ")
        if value is not None:
            zones = ast.literal_eval(value)
    if zones is None:
        raise ValueError(f"no country time zones in the scripts of {root}")
    jan = dt.datetime(2025, 1, 15, 12)
    return {c: int(jan.replace(tzinfo=ZoneInfo(z)).utcoffset()
                   .total_seconds() // 3600) for c, z in zones.items()}


def tz_std_of(icc) -> np.ndarray:
    """Standard-time offset (hours) of each country code; unknown codes raise."""
    icc = pd.Series(np.asarray(icc, dtype=object))
    unknown = sorted(set(icc) - set(TZ_STD))
    if unknown:
        raise KeyError(f"no standard offset for country codes {unknown}")
    return icc.map(TZ_STD).to_numpy(dtype=np.int8)


def match_released(st_hub, st_lon, st_lat, n_hubs: int, nodes: pd.DataFrame,
                   xwalk_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Released station node of each station hub.

    The node (``node_id`` starting with ``STN_``) nearest to any member station
    of the hub, when that distance is at most ``xwalk_m`` (plan step 2: "a
    400 m crosswalk to station hubs"). Returns (released_id, match_m); '' and
    NaN where nothing matches.
    """
    released = np.full(n_hubs, "", dtype=object)
    match_m = np.full(n_hubs, np.nan, dtype=np.float32)
    stn_nodes = nodes[nodes["node_id"].str.startswith("STN_")]
    if len(stn_nodes) == 0 or len(st_hub) == 0:
        return released, match_m
    d = gc_km(np.asarray(st_lon)[:, None], np.asarray(st_lat)[:, None],
              stn_nodes["lon"].to_numpy()[None, :],
              stn_nodes["lat"].to_numpy()[None, :]) * 1000
    best = pd.DataFrame({"hub": np.asarray(st_hub), "node": d.argmin(axis=1),
                         "m": d.min(axis=1)})
    best = best.sort_values(["hub", "m", "node"]).drop_duplicates("hub")
    best = best[best["m"] <= xwalk_m]
    released[best["hub"].to_numpy()] = (
        stn_nodes["node_id"].to_numpy()[best["node"].to_numpy()])
    match_m[best["hub"].to_numpy()] = best["m"].to_numpy()
    return released, match_m


def build_hubs(stations: pd.DataFrame, airports: pd.DataFrame,
               fuas: pd.DataFrame, nodes: pd.DataFrame, radius_m: float,
               xwalk_m: float) -> tuple[pd.DataFrame, pd.Series]:
    """Table ``ps-hubs`` and the map station id -> hub.

    Station hubs (plan, Definitions, "Hub"): the stations of one FUA joined by
    single linkage at ``radius_m``; numbered by their smallest station id.
    A hub is named after its busiest member (column ``calls`` of ``stations``
    when present, else the smallest station id) and placed at the mean of its
    members. Airports follow, in the order of ``airports``.

    stations: stn, name, fua, lon, lat [, calls]; airports: iata, name, icc,
    lon, lat; fuas: id, icc; nodes: node_id, lon, lat (released graph).
    """
    st = stations.sort_values("stn").reset_index(drop=True)
    if "calls" not in st:
        st["calls"] = 0
    lab = single_linkage(st["fua"], st["lon"], st["lat"], radius_m)
    st["hub"] = pd.Series(lab).map(
        st["stn"].groupby(lab).min().rank().astype(np.int64) - 1).to_numpy()
    n_s = int(st["hub"].max()) + 1 if len(st) else 0

    busiest = (st.sort_values(["hub", "calls", "stn"],
                              ascending=[True, False, True])
               .drop_duplicates("hub").set_index("hub")["name"])
    g = st.groupby("hub")
    icc_of_fua = fuas.set_index("id")["icc"]
    s_hubs = pd.DataFrame({
        "hub": np.arange(n_s),
        "kind": "S",
        "name": busiest.reindex(np.arange(n_s)).str.split(" | ", regex=False)
                .str[0].to_numpy(),
        "fua": g["fua"].first().to_numpy(),
        "lon": g["lon"].mean().to_numpy(),
        "lat": g["lat"].mean().to_numpy(),
        "iata": "",
        "stns": [np.sort(x.to_numpy()).astype(np.int32) for _, x in g["stn"]],
    })
    s_hubs["icc"] = s_hubs["fua"].map(icc_of_fua).to_numpy()
    s_hubs["released_id"], s_hubs["match_m"] = match_released(
        st["hub"], st["lon"], st["lat"], n_s, nodes, xwalk_m)

    node_xy = nodes.set_index("node_id")[["lon", "lat"]]
    air_id = "AIR_" + airports["iata"].astype(str)
    found = air_id.isin(node_xy.index).to_numpy()
    node_at = node_xy.reindex(air_id)
    a_hubs = pd.DataFrame({
        "hub": n_s + np.arange(len(airports)),
        "kind": "A",
        "name": airports["name"].to_numpy(),
        "fua": -1,
        "lon": airports["lon"].to_numpy(),
        "lat": airports["lat"].to_numpy(),
        "iata": airports["iata"].to_numpy(),
        "stns": [np.empty(0, dtype=np.int32) for _ in range(len(airports))],
        "icc": airports["icc"].to_numpy(),
        "released_id": np.where(found, air_id.to_numpy(), ""),
        "match_m": np.where(found, gc_km(
            airports["lon"], airports["lat"],
            node_at["lon"].to_numpy(), node_at["lat"].to_numpy()) * 1000, np.nan),
    })
    hubs = pd.concat([s_hubs, a_hubs], ignore_index=True)
    hubs["tz_std"] = tz_std_of(hubs["icc"])
    stn2hub = pd.Series(st["hub"].to_numpy(dtype=np.int32),
                        index=st["stn"].to_numpy(), name="hub")
    return as_contract(hubs, "hubs", S.HUBS), stn2hub


# --- b. Trips ----------------------------------------------------------------
def dateset_masks(datesets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """ic-datesets (``dateset`` + one boolean column per date) -> (dateset ids,
    calendars). The date columns must be exactly the timeline of
    ``pathset.dates``."""
    cols = [c for c in datesets.columns if c != "dateset"]
    if cols != D.date_columns():
        raise ValueError("ic-datesets columns are not the timeline "
                         f"{D.date_columns()[0]}..{D.date_columns()[-1]}")
    return (datesets["dateset"].to_numpy(),
            D.pack_bool(datesets[cols].to_numpy(dtype=bool)))


@njit(cache=True)
def _summer_flags(masks, summer):
    out = np.zeros(masks.shape[0], dtype=np.bool_)
    for i in range(masks.shape[0]):
        n = popcount_row(masks[i])
        out[i] = n > 0 and 2 * and_count(masks[i], summer) >= n
    return out


def summer_flags(masks: np.ndarray) -> np.ndarray:
    """True for calendars with at least half of their active days in EU summer
    time: the season whose offset the upstream export removed from the journey
    (intercity.py lines 98-105 and 315-319, CLOCK_SEASON = journey)."""
    summer = D.pack_bool(D.eu_summer_days()[None, :])[0]
    return _summer_flags(np.ascontiguousarray(masks, dtype=np.uint64), summer)


def local_clock(minute, tz_std, summer) -> np.ndarray:
    """Exported minute -> local wall-clock minute at a hub (plan, Definitions,
    "Trip"): exported minutes + 60 x (standard offset of the hub's country +
    summer flag of the journey)."""
    return (np.asarray(minute, dtype=np.int64)
            + 60 * (np.asarray(tz_std, dtype=np.int64)
                    + np.asarray(summer, dtype=np.int64)))


def _starts(ids: np.ndarray) -> np.ndarray:
    """Offsets of the runs of equal consecutive values, with the end appended."""
    if len(ids) == 0:
        return np.zeros(1, dtype=np.int64)
    return np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True]).astype(np.int64)


_FNV_BASIS = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)
_HASH_SHIFT = 1 << 20          # keeps the hashed minutes positive


@njit(cache=True)
def _hash_calls(start, hub, arr, dep):
    out = np.empty(start.shape[0] - 1, dtype=np.uint64)
    for j in range(out.shape[0]):
        h = _FNV_BASIS
        for r in range(start[j], start[j + 1]):
            h = (h ^ np.uint64(hub[r])) * _FNV_PRIME
            h = (h ^ np.uint64(arr[r] + _HASH_SHIFT)) * _FNV_PRIME
            h = (h ^ np.uint64(dep[r] + _HASH_SHIFT)) * _FNV_PRIME
        out[j] = h
    return out


@njit(cache=True)
def _same_calls(a, b, start, hub, arr, dep):
    n = start[a + 1] - start[a]
    if n != start[b + 1] - start[b]:
        return False
    for t in range(n):
        ra, rb = start[a] + t, start[b] + t
        if hub[ra] != hub[rb] or arr[ra] != arr[rb] or dep[ra] != dep[rb]:
            return False
    return True


@njit(cache=True)
def _representatives(order, h, start, hub, arr, dep):
    """First journey with the same call sequence, for every journey. Journeys
    are visited in ``order`` (hash, then journey index); inside a block of
    equal hashes each one is compared call by call with the representatives
    found so far, so a hash collision cannot merge two sequences."""
    n = order.shape[0]
    rep = np.empty(n, dtype=np.int64)
    reps = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        e = i + 1
        while e < n and h[order[e]] == h[order[i]]:
            e += 1
        n_reps = 0
        for p in range(i, e):
            a = order[p]
            rep[a] = a
            for q in range(n_reps):
                if _same_calls(a, reps[q], start, hub, arr, dep):
                    rep[a] = reps[q]
                    break
            if rep[a] == a:
                reps[n_reps] = a
                n_reps += 1
        i = e
    return rep


def signatures(start, hub, arr, dep) -> np.ndarray:
    """Signature id of each journey: equal for journeys with the same hub
    sequence and the same local arrival and departure minutes (plan,
    Definitions, "Trip"). Ids are numbered by the first journey that carries
    the signature."""
    start, hub, arr, dep = (np.ascontiguousarray(x, dtype=np.int64)
                            for x in (start, hub, arr, dep))
    h = _hash_calls(start, hub, arr, dep)
    order = np.argsort(h, kind="stable")
    rep = _representatives(order, h, start, hub, arr, dep)
    return np.searchsorted(np.unique(rep), rep)


def signature_modes(sig, n_sig: int, rail, route_type) -> tuple[np.ndarray, np.ndarray]:
    """Mode of each signature and a flag for the signatures whose journeys
    disagree.

    R when the line's rail flag is set, else B. Where the journeys of one
    signature carry both flags (the same vehicle described as a train in one
    feed and as a bus in another), the GTFS route type decides: the signature
    takes the class (RAIL_ROUTE_TYPES or BUS_ROUTE_TYPES, as intercity.py
    lines 36-44) of the route type carried by most of its journeys; a tie goes
    to the extended type (>= 100), which says more than the basic 2 or 3, and
    then to the smaller code.
    """
    sig = np.asarray(sig)
    rail = np.asarray(rail, dtype=bool)
    n_rail = np.bincount(sig, weights=rail, minlength=n_sig)
    n_all = np.bincount(sig, minlength=n_sig)
    mode = np.where(n_rail > 0, MODE_R, MODE_B).astype(np.int8)
    mixed = (n_rail > 0) & (n_rail < n_all)
    if mixed.any():
        rows = np.flatnonzero(mixed[sig])
        votes = (pd.DataFrame({"sig": sig[rows],
                               "rt": np.asarray(route_type)[rows]})
                 .value_counts().rename("n").reset_index())
        votes["basic"] = votes["rt"] < 100
        win = (votes.sort_values(["sig", "n", "basic", "rt"],
                                 ascending=[True, False, True, True])
               .drop_duplicates("sig"))
        mode[win["sig"].to_numpy()] = np.where(
            win["rt"].isin(RAIL_ROUTE_TYPES), MODE_R, MODE_B)
    return mode, mixed


@njit(cache=True)
def _or_shifted(group, n_groups, row, shift, masks, n_days):
    out = np.zeros((n_groups, masks.shape[1]), dtype=np.uint64)
    tmp = np.empty(masks.shape[1], dtype=np.uint64)
    for i in range(group.shape[0]):
        shift_row(masks[row[i]], shift[i], tmp, n_days)
        or_into(out[group[i]], tmp)
    return out


def or_calendars(group, n_groups: int, row, shift, masks) -> np.ndarray:
    """Calendar of each group: the OR over its members i of ``masks[row[i]]``
    shifted by ``shift[i]`` days."""
    return _or_shifted(
        np.ascontiguousarray(group, dtype=np.int64), n_groups,
        np.ascontiguousarray(row, dtype=np.int64),
        np.ascontiguousarray(shift, dtype=np.int64),
        np.ascontiguousarray(masks, dtype=np.uint64), D.N_DAYS)


class Trips(NamedTuple):
    trips: pd.DataFrame        # trip, mode, n_jrn, mixed
    calls: pd.DataFrame        # trip, hub, arr, dep (local, normalised)
    masks: np.ndarray          # calendar of each trip (first local departure)
    jrn_trip: pd.Series        # journey id -> trip
    n_calls_merged: int        # station calls absorbed into a hub call


def build_trips(calls: pd.DataFrame, journeys: pd.DataFrame, hub_tz,
                masks: np.ndarray) -> Trips:
    """Trips from journeys (plan, Definitions, "Trip").

    calls: jrn, hub, arr, dep in exported minutes, one row per station call.
    journeys: jrn, rail, route_type, cal (row of ``masks``); every journey of
    ``calls`` must be listed.

    1. Calls are ordered by (jrn, arr); calls with the same arrival minute keep
       the order of the input, which is the order of ic-timetable.
       Consecutive calls of a journey at stations of one hub form one call:
       arrival at the first, departure from the last (a hub is one place:
       "transfers inside a hub need no move"). Upstream lists a station once
       per journey (intercity.py line 274), so after this step a trip calls
       at a hub twice only if it returns to another station of that hub.
    2. Local clock = exported + 60 x (tz_std of the hub + summer flag of the
       journey's calendar).
    3. Normalisation: k = floor(first local departure / 1440); the journey's
       times lose 1440 k and its calendar moves by k days, so a departure at
       24:10 becomes 00:10 on the next day.
    4. Journeys with the same hub sequence and local times are one trip; its
       mode follows ``signature_modes`` and its calendar is the OR of the
       members' calendars, so a copy in a second feed never counts twice.
    """
    jn = journeys.sort_values("jrn").reset_index(drop=True)
    c = calls.sort_values(["jrn", "arr"], kind="stable")
    j_of = np.searchsorted(jn["jrn"].to_numpy(), c["jrn"].to_numpy())
    if (j_of >= len(jn)).any() or (
            jn["jrn"].to_numpy()[np.minimum(j_of, len(jn) - 1)]
            != c["jrn"].to_numpy()).any():
        raise ValueError("calls of journeys that are not listed")
    hub = c["hub"].to_numpy(dtype=np.int64)
    arr = c["arr"].to_numpy(dtype=np.int64)
    dep = c["dep"].to_numpy(dtype=np.int64)

    new = np.r_[True, (j_of[1:] != j_of[:-1]) | (hub[1:] != hub[:-1])]
    first = np.flatnonzero(new)
    last = np.r_[first[1:], len(hub)] - 1
    n_merged = int(len(hub) - len(first))
    j_of, hub, arr, dep = j_of[first], hub[first], arr[first], dep[last]

    present = np.unique(j_of)
    if len(present) != len(jn):
        jn = jn.iloc[present].reset_index(drop=True)
        j_of = np.searchsorted(present, j_of)
    start = _starts(j_of)

    summer = summer_flags(masks)[jn["cal"].to_numpy()]
    tz = np.asarray(hub_tz, dtype=np.int64)[hub]
    arr = local_clock(arr, tz, summer[j_of])
    dep = local_clock(dep, tz, summer[j_of])
    k = dep[start[:-1]] // 1440
    arr -= 1440 * k[j_of]
    dep -= 1440 * k[j_of]

    sig = signatures(start, hub, arr, dep)
    n_sig = int(sig.max()) + 1 if len(sig) else 0
    mode, mixed = signature_modes(sig, n_sig, jn["rail"], jn["route_type"])
    trip_masks = or_calendars(sig, n_sig, jn["cal"].to_numpy(), k, masks)

    rep = np.unique(sig, return_index=True)[1]      # first journey of each trip
    n_calls = (start[1:] - start[:-1])[rep]
    rows = (np.arange(n_calls.sum())
            + np.repeat(start[rep] - (np.cumsum(n_calls) - n_calls), n_calls))
    trip_calls = pd.DataFrame({
        "trip": np.repeat(np.arange(n_sig), n_calls),
        "hub": hub[rows], "arr": arr[rows], "dep": dep[rows]})
    trips = pd.DataFrame({"trip": np.arange(n_sig), "mode": mode,
                          "n_jrn": np.bincount(sig, minlength=n_sig),
                          "mixed": mixed})
    return Trips(trips, trip_calls, trip_masks,
                 pd.Series(sig, index=jn["jrn"].to_numpy(), name="trip"),
                 n_merged)


# --- c. Legs -----------------------------------------------------------------
@njit(cache=True)
def expand_pairs(trip, hub, fua, arr, dep):
    """Expand the ordered calls of each trip into its legs.

    Adapted from pt-links.py:get_timetable_pairs (lines 64-94): the same two
    passes over runs of equal ids, but only pairs whose hubs lie in different
    FUAs are built, and the positions of the departure and of the arrival call
    are returned beside (trip, src, trg, time). ``time`` = arr[j] - dep[i].
    Also returns the number of pairs inside one FUA, which are not built.
    """
    n = len(trip)
    n_pairs = 0
    n_same = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and trip[end] == trip[start]:
            end += 1
        for i in range(start, end - 1):
            for j in range(i + 1, end):
                if fua[i] != fua[j]:
                    n_pairs += 1
                else:
                    n_same += 1
        start = end
    out_trip = np.empty(n_pairs, dtype=trip.dtype)
    out_src = np.empty(n_pairs, dtype=hub.dtype)
    out_trg = np.empty(n_pairs, dtype=hub.dtype)
    out_time = np.empty(n_pairs, dtype=np.int32)
    out_i = np.empty(n_pairs, dtype=np.int64)
    out_j = np.empty(n_pairs, dtype=np.int64)
    k = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and trip[end] == trip[start]:
            end += 1
        for i in range(start, end - 1):
            for j in range(i + 1, end):
                if fua[i] != fua[j]:
                    out_trip[k] = trip[start]
                    out_src[k] = hub[i]
                    out_trg[k] = hub[j]
                    out_time[k] = arr[j] - dep[i]
                    out_i[k] = i
                    out_j[k] = j
                    k += 1
        start = end
    return out_trip, out_src, out_trg, out_time, out_i, out_j, n_same


def build_legs(trip_calls: pd.DataFrame, trip_mode, hub_fua, hub_tz
               ) -> tuple[pd.DataFrame, int]:
    """Legs of the trips: every ordered pair of calls (i before j) whose hubs
    lie in different FUAs (task step c).

    ride  = elapsed minutes from the departure at i to the arrival at j. On
            the local clocks this is arr_j - dep_i less the offset difference
            of the two hubs, which equals the difference of the exported
            minutes.
    dep   = local departure minute at i, reduced modulo 1440.
    shift = floor(dep / 1440): the leg's calendar of departure dates is the
            trip calendar moved by that many days.

    Returns (legs: trip, src, trg, mode, dep, ride, shift; pairs inside one
    FUA, not built).
    """
    trip = trip_calls["trip"].to_numpy(dtype=np.int64)
    hub = trip_calls["hub"].to_numpy(dtype=np.int64)
    tz = 60 * np.asarray(hub_tz, dtype=np.int64)[hub]
    arr = trip_calls["arr"].to_numpy(dtype=np.int64)
    dep = trip_calls["dep"].to_numpy(dtype=np.int64)
    fua = np.asarray(hub_fua, dtype=np.int64)[hub]
    o_trip, o_src, o_trg, o_time, o_i, _, n_same = expand_pairs(
        trip, hub, fua, arr - tz, dep - tz)
    legs = pd.DataFrame({
        "trip": o_trip.astype(np.int32), "src": o_src.astype(np.int32),
        "trg": o_trg.astype(np.int32),
        "mode": np.asarray(trip_mode, dtype=np.int8)[o_trip],
        "dep": (dep[o_i] % 1440).astype(np.int16), "ride": o_time,
        "shift": (dep[o_i] // 1440).astype(np.int16)})
    return legs, int(n_same)


# --- d. Hygiene --------------------------------------------------------------
def hygiene(ride, km, mode, max_ride: int, v_max: dict) -> np.ndarray:
    """Code of the first rule a leg breaks (index in HYGIENE_REASONS, 0 = kept).

    Plan step 1, "Drops": ride <= 0; ride > ``max_ride``; great-circle speed
    above ``v_max[mode letter]`` km/h. A mode without an entry in ``v_max``
    has no speed screen.
    """
    ride = np.asarray(ride, dtype=np.float64)
    mode = np.asarray(mode)
    limit = np.full(len(S.MODES), np.inf)
    for letter, v in v_max.items():
        limit[S.MODES.index(letter)] = v
    with np.errstate(divide="ignore", invalid="ignore"):
        fast = np.asarray(km, dtype=np.float64) * 60 / ride > limit[mode]
    return np.select([ride <= 0, ride > max_ride, fast], [1, 2, 3],
                     default=0).astype(np.int8)


# --- e. Runs -----------------------------------------------------------------
RUN_KEY = ["src", "trg", "mode", "dep", "ride"]


def merge_equal(legs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Runs: legs with equal (src, trg, mode, dep, ride) are one run (task
    step e). The run carries the smallest trip key of its legs; this key is
    provisional, ``vehicle_keys`` sets the final one. Returns the runs sorted
    by (src, trg, mode, dep, ride) and, for every leg, the row of its run. The
    run's calendar is the OR of its legs' calendars (``or_calendars`` or
    ``weekday_calendars``)."""
    cols = [legs[c].to_numpy() for c in RUN_KEY]
    order = np.lexsort([legs["trip"].to_numpy()] + cols[::-1])
    new = np.zeros(len(legs), dtype=bool)
    new[:1] = True
    for col in cols:
        s = col[order]
        new[1:] |= s[1:] != s[:-1]
    run_of_leg = np.empty(len(legs), dtype=np.int64)
    run_of_leg[order] = np.cumsum(new) - 1
    runs = legs.iloc[order[new]][RUN_KEY + ["trip"]].reset_index(drop=True)
    return runs, run_of_leg


@njit(cache=True)
def _fits(s, r, dep, ride, lo, hi, tol):
    """Kept run s can take run r: s's departure and ride lie within ``tol``
    of every member of r (bounds lo/hi: dep, ride)."""
    return (abs(dep[s] - lo[r, 0]) <= tol and abs(dep[s] - hi[r, 0]) <= tol
            and abs(ride[s] - lo[r, 1]) <= tol
            and abs(ride[s] - hi[r, 1]) <= tol)


@njit(cache=True)
def _near_keepers(group, dep, ride, lo, hi, n_dates, masks, tol):
    n = group.shape[0]
    keeper = np.full(n, -1, dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    a = 0
    while a < n:
        b = a + 1
        while b < n and group[b] == group[a]:
            b += 1
        order = a + np.argsort(-n_dates[a:b], kind="mergesort")
        for i in range(b - a):
            rank[order[i]] = i
        for r in order:
            best = -1
            s = r - 1
            while s >= a and dep[r] - dep[s] <= tol:
                if (keeper[s] == s and _fits(s, r, dep, ride, lo, hi, tol)
                        and (best < 0 or rank[s] < rank[best])
                        and and_any(masks[s], masks[r])):
                    best = s
                s -= 1
            s = r + 1
            while s < b and dep[s] - dep[r] <= tol:
                if (keeper[s] == s and _fits(s, r, dep, ride, lo, hi, tol)
                        and (best < 0 or rank[s] < rank[best])
                        and and_any(masks[s], masks[r])):
                    best = s
                s += 1
            if best < 0:
                keeper[r] = r
            else:
                keeper[r] = best
                or_into(masks[best], masks[r])
        a = b
    return keeper


def merge_near(runs: pd.DataFrame, masks: np.ndarray, tol: int
               ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Merge near-duplicate runs (plan, Definitions, "Near-duplicate").

    Two runs of one (src, trg, mode) are near-duplicates when their departure
    minutes and their riding times both differ by at most ``tol`` and their
    calendars share a date. The rule for chains, made deterministic:

    - the runs of a hub pair and mode are visited by decreasing number of
      dates (ties: earlier departure, then shorter ride);
    - a visited run is absorbed by a run kept earlier when it is a
      near-duplicate of it, compared with the kept run's own departure and
      riding time and with its calendar as grown so far; among several such
      runs the one visited first (the one with more dates) takes it;
    - otherwise the run is kept.

    A kept run keeps its departure, riding time and (provisional) trip key and
    gets the OR of the calendars it absorbed. Because the comparison is always
    with the kept run itself, an absorbed run never lies more than ``tol``
    from the times it is filed under, however long the chain. Departures are
    compared as minutes of the day, so 23:59 and 00:00 are not
    near-duplicates.

    A kept calendar that grows can come to share a date with a run kept
    before it grew, so the pass is repeated on its own output until no run is
    absorbed (a fixed point). Each pass applies the same rule to the merged
    runs, with their grown calendars and date counts, and one more condition
    that keeps the guarantee above: a merged run is absorbed only when the
    taker's departure and riding time lie within ``tol`` of every input run
    it holds. Near-duplicates left at the fixed point are pairs that could
    merge only by moving an input run more than ``tol``.

    ``runs`` must be sorted by (src, trg, mode, dep, ride) with ``masks``
    row-aligned. Returns (kept runs with ``n_dates``, their calendars, row of
    the kept run for every input run).
    """
    out = runs.reset_index(drop=True)
    out_masks = np.array(masks, dtype=np.uint64, order="C")
    times = out[["dep", "ride"]].to_numpy(dtype=np.int64)
    lo, hi = times.copy(), times.copy()
    kept_of = np.arange(len(out))
    while True:
        key = [out[c].to_numpy() for c in RUN_KEY[:3]]
        new = np.zeros(len(out), dtype=bool)
        new[:1] = True
        for col in key:
            new[1:] |= col[1:] != col[:-1]
        keeper = _near_keepers(
            np.cumsum(new), out["dep"].to_numpy(dtype=np.int64),
            out["ride"].to_numpy(dtype=np.int64), lo, hi,
            D.popcount_rows(out_masks).astype(np.int64), out_masks, tol)
        kept = keeper == np.arange(len(out))
        kept_of = (np.cumsum(kept) - 1)[keeper][kept_of]
        out = out[kept].reset_index(drop=True)
        out_masks = np.ascontiguousarray(out_masks[kept])
        if kept.all():
            break
        lo = np.full((len(out), 2), np.iinfo(np.int64).max)
        hi = np.full((len(out), 2), np.iinfo(np.int64).min)
        np.minimum.at(lo, kept_of, times)
        np.maximum.at(hi, kept_of, times)
    out["n_dates"] = D.popcount_rows(out_masks)
    return out, out_masks, kept_of


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _component_min(trip, event, n_trips, n_events):
    parent = np.arange(n_trips)
    first = np.full(n_events, -1, dtype=np.int64)
    for k in range(trip.shape[0]):
        t, e = trip[k], event[k]
        if first[e] < 0:
            first[e] = t
        else:
            a, b = _find(parent, first[e]), _find(parent, t)
            if a < b:
                parent[b] = a
            elif b < a:
                parent[a] = b
    key = np.full(n_events, -1, dtype=np.int64)
    for e in range(n_events):
        if first[e] >= 0:
            key[e] = _find(parent, first[e])
    return key


def vehicle_keys(trip, event, n_events: int) -> np.ndarray:
    """Final trip key of every event (contract ``ps-events``: "equal for all
    legs of one vehicle run pattern").

    ``trip`` and ``event`` are row-aligned: leg (or flight record) k belongs
    to trip ``trip[k]`` and was merged into event ``event[k]`` (-1 = not
    written). Merging equal legs and near-duplicates puts legs of several
    trips into one event, so one trip's legs can land in events that the
    merge rules key differently. Trips that share an event are joined
    (union-find); every event gets the smallest trip id of its group, so all
    legs of a trip carry one key and the transfer rule "trip keys differ"
    (transfer.py) never sees a vehicle staying put as a transfer.

    One key per event cannot do better: the legs of a trip must share a key,
    so must the trips of an event, which makes the groups the connected
    components. Two vehicles of one group that never share an event also
    share the key, so a transfer between them is not admitted.
    Raises if a written event has no leg.
    """
    trip = np.asarray(trip, dtype=np.int64)
    event = np.asarray(event, dtype=np.int64)
    on = event >= 0
    n_trips = int(trip.max()) + 1 if len(trip) else 0
    key = _component_min(trip[on], event[on], n_trips, int(n_events))
    if (key < 0).any():
        raise ValueError(f"{int((key < 0).sum())} events without a leg")
    return key


# --- f. Flights --------------------------------------------------------------
def check_air_alignment(timetable: pd.DataFrame, schedule: pd.DataFrame) -> None:
    """air-schedule must be row-aligned with the filtered air-timetable (same
    number of rows, same src, trg, carrier, flight and start_date in every
    row); the riding time is read from it by position. Raises ValueError
    otherwise."""
    if len(timetable) != len(schedule):
        raise ValueError(f"air-schedule has {len(schedule):,} rows, the "
                         f"filtered air-timetable {len(timetable):,}")
    for col in ["src", "trg", "carrier", "flight", "start_date"]:
        a = timetable[col].astype(str).to_numpy()
        b = schedule[col].astype(str).to_numpy()
        if (a != b).any():
            raise ValueError(f"air-schedule is not row-aligned with "
                             f"air-timetable: column {col} differs in "
                             f"{int((a != b).sum()):,} rows")


def op_days_bits(op_days) -> np.ndarray:
    """OAG days of operation -> weekday bits (bit 0 = Monday). The string
    lists the digits of the operating weekdays, '1' = Monday ... '7' = Sunday,
    blanks elsewhere (air-times.py lines 142-144)."""
    codes, uniq = pd.factorize(pd.Series(op_days).astype(str))
    bits = np.array([sum(1 << (int(ch) - 1) for ch in set(s.replace(" ", "")))
                     for s in uniq], dtype=np.uint8)
    return bits[codes]


def day_range(start_date, end_date) -> tuple[np.ndarray, np.ndarray]:
    """First and last timeline day of each date range (ISO date strings),
    clipped to the timeline; first > last when nothing is left."""
    def index(col, lo, hi):
        codes, uniq = pd.factorize(pd.Series(col).astype(str))
        days = (pd.to_datetime(uniq) - pd.Timestamp(D.DAY0)).days.to_numpy()
        return np.clip(days, lo, hi)[codes].astype(np.int64)
    return (index(start_date, 0, D.N_DAYS),
            index(end_date, -1, D.N_DAYS - 1))


@njit(cache=True)
def _weekday_masks(group, n_groups, day0, day1, bits, weekday0, n_words):
    out = np.zeros((n_groups, n_words), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(group.shape[0]):
        for d in range(day0[i], day1[i] + 1):
            if (bits[i] >> ((d + weekday0) % 7)) & 1:
                out[group[i], d >> 6] |= one << np.uint64(d & 63)
    return out


def weekday_calendars(group, n_groups: int, day0, day1, bits) -> np.ndarray:
    """Calendar of each group: the OR over its members of the days day0..day1
    whose weekday is set in ``bits`` (task step f)."""
    return _weekday_masks(
        np.ascontiguousarray(group, dtype=np.int64), n_groups,
        np.ascontiguousarray(day0, dtype=np.int64),
        np.ascontiguousarray(day1, dtype=np.int64),
        np.ascontiguousarray(bits, dtype=np.uint8), D.WEEKDAY0, D.N_WORDS)


# --- g, h. Classes, links, events ---------------------------------------------
@njit(cache=True)
def _classes(new_group, ride, one_class, abs_min, rel):
    cls = np.empty(ride.shape[0], dtype=np.int16)
    c = 0
    anchor = 0.0
    for i in range(ride.shape[0]):
        if new_group[i]:
            c = 0
            anchor = ride[i]
        elif not one_class[i] and ride[i] > anchor + max(abs_min, rel * anchor):
            c += 1
            anchor = ride[i]
        cls[i] = c
    return cls


def ride_classes(runs: pd.DataFrame, abs_min: float, rel: float,
                 air_one_class: bool) -> np.ndarray:
    """Riding-time class of each run within its (src, trg, mode); 0 = fastest.

    Plan, Definitions, "Service link" (Cats 2011 pp. 74, 80: lines are pooled
    only on equal riding time): the distinct rides are sorted; a class is
    anchored at the fastest ride not yet assigned and takes every ride within
    max(``abs_min``, ``rel`` x anchor) of the anchor. With ``air_one_class``
    all flights of an airport pair form one class.
    """
    cols = [runs[c].to_numpy() for c in ["src", "trg", "mode", "ride"]]
    order = np.lexsort(cols[::-1])
    new = np.zeros(len(runs), dtype=bool)
    new[:1] = True
    for col in cols[:3]:
        s = col[order]
        new[1:] |= s[1:] != s[:-1]
    one = (cols[2][order] == MODE_A) & bool(air_one_class)
    cls = np.empty(len(runs), dtype=np.int16)
    cls[order] = _classes(new, cols[3][order].astype(np.float64), one,
                          float(abs_min), float(rel))
    return cls


@njit(cache=True)
def add_daily_counts(mask, count):
    """Add 1 to ``count[d]`` for every day d set in the calendar."""
    for w in range(mask.shape[0]):
        x = mask[w]
        d = w * 64
        while x:
            if x & np.uint64(1):
                count[d] += 1
            x >>= np.uint64(1)
            d += 1


@njit(cache=True)
def median_active(count):
    """(median of the positive entries, number of them, largest entry)."""
    pos = np.sort(count[count > 0])
    m = pos.shape[0]
    if m == 0:
        return 0.0, 0, 0
    return (pos[(m - 1) // 2] + pos[m // 2]) / 2.0, m, pos[m - 1]


@njit(cache=True)
def _link_stats(ev0, ev1, ride, masks, n_days, fixed0, fixed1):
    n = ev0.shape[0]
    ride_mean = np.empty(n, dtype=np.float64)
    freq = np.empty(n, dtype=np.float64)
    freq_fixed = np.empty(n, dtype=np.float64)
    ints = np.empty((n, 7), dtype=np.int64)
    count = np.zeros(n_days, dtype=np.int32)
    for l in range(n):
        count[:] = 0
        weighted = 0
        lo, hi = ride[ev0[l]], ride[ev0[l]]
        for e in range(ev0[l], ev1[l]):
            add_daily_counts(masks[e], count)
            weighted += ride[e] * popcount_row(masks[e])
            lo, hi = min(lo, ride[e]), max(hi, ride[e])
        freq[l], active, freq_max = median_active(count)
        days = np.flatnonzero(count)
        total = count.sum()
        ride_mean[l] = weighted / total
        freq_fixed[l] = count[fixed0:fixed1 + 1].sum() / (fixed1 - fixed0 + 1)
        ints[l, 0], ints[l, 1] = lo, hi
        ints[l, 2], ints[l, 3], ints[l, 4] = active, days[0], days[-1]
        ints[l, 5], ints[l, 6] = freq_max, total
    return ride_mean, freq, freq_fixed, ints


class Network(NamedTuple):
    links: pd.DataFrame        # ps-links
    sparse: pd.DataFrame       # ps-sparse
    events: pd.DataFrame       # ps-events
    masks: np.ndarray          # ps-event-masks
    event_of_run: np.ndarray   # row of ps-events for every input run, -1 if none


def build_links(runs: pd.DataFrame, masks: np.ndarray, cls, hubs: pd.DataFrame,
                min_active: int, fixed_days: tuple[int, int]) -> Network:
    """Service links, their events and the sparse links (task steps g, h).

    A link is (src, trg, mode, class). Its attributes (contract ``ps-links``):
    ride = mean riding time of its events weighted by their dates;
    active_dates = days in the OR of the event calendars; freq = median over
    the active dates of the departures that date (plan, "Link frequency";
    upstream pt-links.py lines 110-146); freq_max = most departures on one
    date; freq_span = departures / days from first to last active day;
    freq_fixed = departures inside ``fixed_days`` / days of that period (deck
    slide 237); active_share = active dates / days from first to last.

    Links with fewer than ``min_active`` active dates go to the sparse table
    (numbered on their own, 0..S-1) and their events are not kept; ``cls``
    keeps its value from before that split. Events are sorted by (link, dep,
    ride).

    runs: src, trg, mode, dep, ride, trip, n_dates with ``masks`` row-aligned;
    hubs: ps-hubs.
    """
    if (runs["n_dates"].to_numpy() <= 0).any():
        raise ValueError("runs without a date")
    cols = [runs["src"].to_numpy(), runs["trg"].to_numpy(),
            runs["mode"].to_numpy(), np.asarray(cls)]
    order = np.lexsort([runs["ride"].to_numpy(), runs["dep"].to_numpy()]
                       + cols[::-1])
    new = np.zeros(len(runs), dtype=bool)
    new[:1] = True
    for col in cols:
        s = col[order]
        new[1:] |= s[1:] != s[:-1]
    ev = runs.iloc[order].reset_index(drop=True)
    ev_masks = np.ascontiguousarray(masks[order])
    ev0 = np.flatnonzero(new)
    ev1 = np.r_[ev0[1:], len(ev)]
    ride_mean, freq, freq_fixed, ints = _link_stats(
        ev0, ev1, ev["ride"].to_numpy(dtype=np.int64), ev_masks, D.N_DAYS,
        fixed_days[0], fixed_days[1])
    src, trg = ev["src"].to_numpy()[ev0], ev["trg"].to_numpy()[ev0]
    tz, lon, lat = (hubs[c].to_numpy() for c in ["tz_std", "lon", "lat"])
    span = ints[:, 4] - ints[:, 3] + 1
    links = pd.DataFrame({
        "src": src, "trg": trg,
        "mode": MODE_LETTERS[ev["mode"].to_numpy()[ev0]],
        "cls": np.asarray(cls)[order][ev0],
        "ride": ride_mean, "ride_min": ints[:, 0], "ride_max": ints[:, 1],
        "tz_shift": 60 * (tz[trg].astype(np.int64) - tz[src]),
        "km": gc_km(lon[src], lat[src], lon[trg], lat[trg]),
        "n_events": ev1 - ev0, "active_dates": ints[:, 2],
        "first_day": ints[:, 3], "last_day": ints[:, 4],
        "freq": freq, "freq_max": ints[:, 5], "freq_span": ints[:, 6] / span,
        "freq_fixed": freq_fixed, "active_share": ints[:, 2] / span})

    keep = links["active_dates"].to_numpy() >= min_active
    link_id = np.where(keep, np.cumsum(keep) - 1, -1)
    ev_link = np.repeat(link_id, ev1 - ev0)
    ev_keep = ev_link >= 0

    kept = links[keep].copy()
    kept["link"] = np.arange(len(kept))
    kept["ev1"] = np.cumsum(kept["n_events"].to_numpy())
    kept["ev0"] = kept["ev1"] - kept["n_events"]
    sparse = links[~keep].copy()
    sparse["link"] = np.arange(len(sparse))

    events = ev[ev_keep].copy()
    events["link"] = ev_link[ev_keep]
    event_of_sorted = np.where(ev_keep, np.cumsum(ev_keep) - 1, -1)
    event_of_run = np.empty(len(runs), dtype=np.int64)
    event_of_run[order] = event_of_sorted
    return Network(
        as_contract(kept, "links", S.LINKS),
        as_contract(sparse, "links",
                    [c for c in S.LINKS if c not in ("ev0", "ev1")]),
        as_contract(events, "events", S.EVENTS),
        ev_masks[ev_keep], event_of_run)


def link_members(event, link_of_event, agency, route_type, line) -> pd.DataFrame:
    """Table ``ps-link-members``: for every (link, agency, route type, line)
    the number of the link's events it contributes to.

    The arguments are row-aligned contributions: one row says that the event
    was (also) run by this agency on this line. A contribution listed twice
    counts once.
    """
    m = pd.DataFrame({"event": np.asarray(event), "agency": np.asarray(agency),
                      "route_type": np.asarray(route_type),
                      "line": np.asarray(line)}).drop_duplicates()
    m["link"] = np.asarray(link_of_event)[m["event"].to_numpy()]
    m = (m.groupby(["link", "agency", "route_type", "line"], sort=True)
         .size().rename("n_events").reset_index())
    return as_contract(m, "members", S.LINK_MEMBERS)


def drops_table(rows: list[tuple]) -> pd.DataFrame:
    """Table ``ps-drops`` from (step, reason, n, detail) tuples."""
    return as_contract(pd.DataFrame(rows, columns=S.DROPS), "drops", S.DROPS)


# --- i. Car ------------------------------------------------------------------
# Tolerance (minutes) for recognising the unit bug in car-links. Safe both
# ways: float32 rounding of times up to a few thousand minutes, summed over
# three values, stays below 1e-3 min; the smallest lost crossing share, 1/60
# of the 49 min Reggio di Calabria -> Messina crossing, is 0.82 min.
_CAR_TOL_MIN = 0.05


def crossing_sides(t: pd.Series, a: int, b: int, minutes_ab: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """FUAs on the road network of ``a`` and on that of ``b``, the two ends of
    a fixed crossing, read from the car table ``t`` (index (src, trg),
    minutes).

    car-times.py lines 44-50 route every pair across the water through the
    crossing, so with the unit bug every FUA y on a's side has minutes(y, b) =
    minutes(y, a) + crossing / 60, and every FUA x on b's side has
    minutes(a, x) = minutes(b, x) + crossing / 60. No FUA of another road
    network fits either equation.
    """
    to_a, to_b = t.xs(a, level=1), t.xs(b, level=1)
    from_a, from_b = t.xs(a, level=0), t.xs(b, level=0)
    on_a = (to_b - to_a - minutes_ab / 60).abs() < _CAR_TOL_MIN
    on_b = (from_a - from_b - minutes_ab / 60).abs() < _CAR_TOL_MIN
    return on_a[on_a].index.to_numpy(), on_b[on_b].index.to_numpy()


def build_car(car: pd.DataFrame, fua_names: pd.Series,
              crossings=CAR_CROSSINGS) -> pd.DataFrame:
    """Table ``ps-car``: the off-diagonal rows of car-links, with the crossing
    restored on the pairs that use a fixed connection across water.

    car-times.py adds the crossing constants, given in km and minutes (lines
    33-39), to OSRM values in metres and seconds (lines 49-50, 57-58) and then
    divides everything by 1000 and by 60 (lines 66-67). A pair across the
    Channel, the Strait of Messina or the Gulf of Finland therefore holds only
    1/60 of the crossing minutes and 1/1000 of its km. The affected rows are
    the pairs from one side of a crossing to the other (``crossing_sides``);
    they get back 59/60 of the minutes and 999/1000 of the km, and ``fixed`` =
    True.

    Each such row is first checked against the way it was built: minutes(y, x)
    = minutes(y, A) + minutes(B, x) + crossing / 60. A crossing without such
    rows, or a row that does not fit, raises, so a corrected upstream table is
    not corrected twice.

    car: src_fua, trg_fua, time (min), dist (km), with the diagonal;
    fua_names: FUA id -> name.
    """
    t = car.set_index(["src_fua", "trg_fua"])["time"].astype(np.float64)
    out = car[car["src_fua"] != car["trg_fua"]].copy()
    out["time"] = out["time"].astype(np.float64)
    out["dist"] = out["dist"].astype(np.float64)
    out["fixed"] = False
    for city_a, city_b, km_ab, min_ab, km_ba, min_ba in crossings:
        a = int(fua_names[fua_names == city_a].index[0])
        b = int(fua_names[fua_names == city_b].index[0])
        side_a, side_b = crossing_sides(t, a, b, min_ab)
        for p, q, side_p, side_q, km, minutes in [
                (a, b, side_a, side_b, km_ab, min_ab),
                (b, a, side_b, side_a, km_ba, min_ba)]:
            rows = (out["src_fua"].isin(side_p) & out["trg_fua"].isin(side_q))
            y, x = out.loc[rows, "src_fua"], out.loc[rows, "trg_fua"]
            built = (
                t.reindex(pd.MultiIndex.from_arrays([y, np.full(len(y), p)])).to_numpy()
                + t.reindex(pd.MultiIndex.from_arrays([np.full(len(x), q), x])).to_numpy()
                + minutes / 60)
            gap = np.abs(out.loc[rows, "time"].to_numpy() - built)
            if len(gap) == 0 or not (gap < _CAR_TOL_MIN).all():
                raise ValueError(
                    f"car-links rows {fua_names[p]} -> {fua_names[q]} do not "
                    "carry the unit bug of car-times.py lines 49-67")
            out.loc[rows, "time"] += minutes * (1 - 1 / 60)
            out.loc[rows, "dist"] += km * (1 - 1 / 1000)
            out.loc[rows, "fixed"] = True
    out = out.rename(columns={"src_fua": "o", "trg_fua": "d",
                              "time": "minutes", "dist": "km"})
    return as_contract(out.sort_values(["o", "d"]), "car", S.CAR)


def car_speed(car: pd.DataFrame, fua, lon, lat) -> np.ndarray:
    """Straight-line speed (km/h) of each ps-car row between the FUA centres."""
    xy = pd.DataFrame({"lon": np.asarray(lon), "lat": np.asarray(lat)},
                      index=np.asarray(fua))
    o, d = xy.reindex(car["o"]), xy.reindex(car["d"])
    km = gc_km(o["lon"].to_numpy(), o["lat"].to_numpy(),
               d["lon"].to_numpy(), d["lat"].to_numpy())
    return km * 60 / car["minutes"].to_numpy(dtype=np.float64)
