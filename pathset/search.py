"""Candidate paths of every FUA pair, and the rules per pair.

Plan steps 3 and 4 and "Method in brief"; contract: the "Paths" part of
``pathset/schemas.py``. Everything is a function of a ``Net`` (made by
``build_network`` from the tables of the contract) and a parameter dict.

Path    p = [e0, e1 .. en, en+1]: car access from the origin FUA o to the first
        hub (ps-connectors.acc_min), n = 1 .. MAX_TRANSFERS + 1 service links,
        car egress from the last hub to the destination FUA d (egr_min). Links
        k and k + 1 are a row of ps-transfers (link_in, link_out); the alight
        hub of k and the board hub of k + 1 differ when that row has a move.
City    of a station hub: its ps-hubs.fua. Of an airport: the fua of its
        ps-connectors row with the least acc_min (ties: lower fua); an airport
        without such a row is its own city, -1 - hub (plan, Definitions,
        "Hub"). c_first, c_last: the cities of the first and the last hub; the
        cities of transfer k: {city(alight hub of k), city(board hub of k+1)}.
R0      every clause holds: o != d; c_first != c_last; c_first != d;
        c_last != o; the hubs of the path pairwise distinct; no city of a
        transfer is o, d, c_first or c_last; the cities of transfers 1 and 2
        are disjoint; the mode sequence is a word of MODE_WORDS
        (``word_allowed``); the chain exists (one link: always, ps-links holds
        only links active on >= MIN_ACTIVE_DATES dates; two links: the
        ps-transfers row has n_days >= MIN_ACTIVE_DATES on basis 'D' or >= 1
        on 'T'; three links: both rows exist and transfer.triple_stats passes
        the same test, on typical days when a transfer joins air and ground);
        T <= T_MAX_MIN. Plan step 3, "Generation". Distinct hubs follow from
        the city clauses (a hub repeated in two places of a path repeats a
        city those clauses keep apart); the search checks them anyway.
Values  rides from ps-links; transfer minutes ft (f_trans) and the counted
        chain frequency from ps-links.freq (one link), the ps-transfers row
        (two) or triple_stats (three). freq = cost.k_capped_frequency of the
        count: the count, at most the sparsest link's freq_max (pathset.cost,
        module docstring); it enters hwt, gtc, the rules and the column freq,
        while freq_counted keeps the count. T, gtt, hwt, gtc and the seven
        components come from the cost kernels; hwt_first is the hidden wait at
        the first link's ps-links.freq (the deck's form, slide 228). basis and
        n_days of a one-link chain: 'D' and its active_dates.
R3      the only cut during generation (plan step 4): a candidate is cut iff
        gtt > (1 + MU) x the least gtt of its mode class AND gtc > (1 + MU) x
        the least gtc of its mode class, both minima over all R0 candidates of
        the pair. Values are compared as stored (float32), which is how
        rules.apply_rules sees them, so that its own R3 cuts nothing more.

Search, one origin FUA ("shard") at a time, in compiled loops over first,
second and third service link ("chains") and the destinations served by the
last hub (plan step 3):
  pass 1  the exact least gtt per (d, mode sequence) and the candidate that
          has it (ps-best-by-sequence; ties: lower l1, l2, l3), the least gtc
          per (d, mode class) and the least T per d.
  pass 2  per (d, sequence) the R0 candidates and the ones R3 keeps; the
          identities of the kept ones go into a buffer that grows by chunks
          of CHUNK rows (ID_BYTES each; never truncated), with the
          triple_stats values of the kept three-link chains (TV_BYTES per
          chain).
Then the rows of the kept candidates are computed exactly from the buffer, by
destination block: a block holds at most MEM_CAND_GB, less the buffer, of rows
estimated at CAND_BYTES each (plan step 3: never truncate; a destination above
the limit is a block of its own). No chain is walked twice.
One- and two-link chains are exact and cheap. A three-link chain calls
transfer.triple_stats only when, for at least one destination its last hub
serves, its lower bounds (cost.k_bounds with transfer times lo + move of the
window, never a pair's ft_min, and the links' freq_max) are not above the
current minimum (pass 1) or inside the R3 margin (pass 2). Bounds only skip
work, they never stand in for a value. triple_stats results are kept per chain
in a cache (passes 1 and 2) that is emptied whenever its entries, at
CACHE_BYTES each, and the buffer would exceed MEM_CAND_GB, and after pass 2;
a result is recomputed identically, so emptying never changes the output.
Every row computed exactly is checked against its bounds with
cost.k_bounds_hold; a violation stops the shard (``BoundViolation``;
``BOUND_TEST_SHIFT`` raises the bounds for tests).

Memory of a shard beyond the network and the transfer workspace: at most about
MEM_CAND_GB, unless the identity buffer alone is larger (ID_BYTES per kept
candidate, kept whole until the rows of the last block are made).

Counting convention of the funnel's n_r0: a three-link chain whose bounds lie
outside the R3 margin for every destination its last hub serves is never
evaluated, so its existence and its T are unknown. It counts as an R0
candidate of every destination for which it passes the other clauses and the
lower bound of its T is at most T_MAX_MIN. n_r0 is therefore an upper bound
of the true count; the trace of a named pair (MU = inf) is exact, and so is
n_r0 of a run at MU = inf. The search log (funnel attrs["search"], counter
r0_by_bounds) holds how many of a shard's n_r0 were counted by bounds.

The car (INCLUDE_CAR) is one more candidate of every pair with a ps-car row:
seq 'C', n 0, T = minutes, gtt = gtc = BETA_IVT['C'] x minutes, hwt 0, freq
inf, key 'car', no hubs (cities ''). It is counted in every funnel column and
in ps-seq-counts (seq 'C'), and it is the row 'C' of ps-best-by-sequence.

The compiled functions reach the cost kernels and are never cached on disk
(pathset.cost, "Compiled callers"). Their names begin with ``_srch_`` so that
the name-based scan of tests/test_cost.py cannot mix them up with names used
elsewhere.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from typing import NamedTuple

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import Dict, List

from . import cost as C
from . import io
from . import rules
from . import schemas as S
from . import transfer as TR

# --- Mode sequences ------------------------------------------------------------
# Words of one to three links over A, B, R; word id = position. With the mode
# codes A = 0, B = 1, R = 2 the id of m1 is m1, of m1 m2 is 3 + 3 m1 + m2 and
# of m1 m2 m3 is 12 + 9 m1 + 3 m2 + m3 (compiled into the search).
assert [C.MODE_CODE[m] for m in S.MODES] == [0, 1, 2]
WORDS = tuple("".join(w) for n in range(1, C.MAX_LINKS + 1)
              for w in itertools.product(S.MODES, repeat=n))
N_WORDS = len(WORDS)
WORD_OF = {w: i for i, w in enumerate(WORDS)}
N_CLASSES = 1 << len(S.MODES)    # a mode class is a bit set of A, B, R
MODE_WORD_RULES = ("air_block_max2", "deck35", "oct21")
OCT21_OUT = ("ABB", "ABR", "ARB", "ARR", "BBA", "BRA", "RBA", "RRA")
AIR = C.MODE_CODE["A"]

# --- Kernel constants ----------------------------------------------------------
MIN, COLLECT = 0, 1                      # what a walk does: pass 1, pass 2
NO_CITY = -(2 ** 62)                     # unused transfer city
BASIS_D = TR.BASIS_D
BASIS_T = TR.BASIS_T
P_TMAX, P_ACTIVE, P_COMMON, P_MAXTR, P_BUDGET = 0, 1, 2, 3, 4   # layout of Fixed.prm
# counters of a walk: chains of 1, 2, 3 links visited, R0 candidates with
# exact values, triple_stats calls, results reused, three-link chains not
# evaluated (bounds, or no destination left), R0 candidates counted by their
# bounds, identities written, triple values written, times the cache was
# emptied
(ST_CH1, ST_CH2, ST_CH3, ST_CAND, ST_CALLS, ST_HITS, ST_SKIP, ST_R0B, ST_FILL,
 ST_TV, ST_CLEAR) = range(11)
N_STATS = 11
STAT_NAMES = ("chains1", "chains2", "chains3", "evaluated", "triple_calls",
              "triple_reused", "chains3_skipped", "r0_by_bounds", "written",
              "triple_rows", "cache_cleared")

# Buffer of the kept candidates: rows (d, l1, l2, l3, q1, q2, row of the
# triple values or -1) and rows (basis, n_days, counted freq, ft1, ft2) of the
# kept three-link chains, in chunks of CHUNK rows (read when a shard starts)
ID_COLS, TV_COLS = 7, 5
ID_BYTES, TV_BYTES = 4 * ID_COLS, 8 * TV_COLS
CHUNK = 1 << 16
# Bytes of one triple_stats cache entry (numba Dict int64 -> 5 float64): 70 to
# 93 measured; rounded up
CACHE_BYTES = 96

# Float columns of a candidate row, in the order of the fill kernel
FLOATS = (("acc", "egr", "mv1", "mv2", "ft1", "ft2") + C.COMPONENTS
          + ("T", "gtt", "hwt", "hwt_first", "gtc", "freq", "freq_counted"))
N_F = len(FLOATS)
N_TOK = 2 * C.MAX_LINKS                  # hubs of a path, at most
N_I = 6 + N_TOK                          # n, word, h_first, h_last, basis, n_days, tokens

# Estimated peak bytes of one kept candidate while a destination block is
# turned into rows (identity, kernel output, DataFrame with its strings). On a
# synthetic shard of 441,775 candidates the PATH table took 174 bytes a row and
# the peak working set grew by about 490 bytes a row; rounded up
CAND_BYTES = 600

# Test hook: added to every lower bound in the check of the exact rows; a
# positive value makes the check fail (tests/test_search.py)
BOUND_TEST_SHIFT = 0.0


class BoundViolation(RuntimeError):
    """A lower bound of the search exceeds the exact value of a candidate."""


# --- Contract tables -----------------------------------------------------------
TABLES = {"path": S.PATH, "alternatives": S.ALTERNATIVES,
          "variants": S.VARIANTS, "best": S.BEST_BY_SEQUENCE,
          "funnel": S.FUNNEL, "seq_counts": S.SEQ_COUNTS,
          "unserved": S.UNSERVED, "trace": S.TRACE}
TYPES = {"o": "int16", "d": "int16", "seq": "str", "n": "int8",
         "h_first": "int32", "h_last": "int32", "l1": "int32", "l2": "int32",
         "l3": "int32", "cities": "str", "key": "str", "basis": "str",
         "n_days": "int32", **{c: "float32" for c in FLOATS},
         "rank": "int16", "member": "int16", "cls": "str",
         "freq_pooled": "float32", "n_variants": "int32", "status": "str",
         **{c: "int64" for c in ("n_r0", "n_r3", "n_r1", "n_r2", "n_kept",
                                 "n_capped", "cand", "alt", "by")},
         "n_classes": "int8", "best_T": "float32", "best_gtt": "float32",
         "car_min": "float32", "pt": "bool", "reason": "str", "rule": "str",
         "detail": "str"}


def as_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Columns of a path table (``TABLES``) in contract order and dtypes."""
    cols = TABLES[table]
    return df[cols].astype({c: TYPES[c] for c in cols}).reset_index(drop=True)


def empty_table(table: str) -> pd.DataFrame:
    return as_table(pd.DataFrame({c: [] for c in TABLES[table]}), table)


def _concat(frames: list, table: str) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return empty_table(table)
    return as_table(pd.concat(frames, ignore_index=True), table)


# --- Mode words and classes ----------------------------------------------------
def word_allowed(word: str, rule: str) -> bool:
    """Whether a mode sequence is allowed under the rule MODE_WORDS.

    air_block_max2: the A's form one block and there are at most two (36 words
    of one to three links); deck35: the same without RAR (the 35 sequences of
    the deck's path set, slide 238); oct21: the same without ABB, ABR, ARB,
    ARR, BBA, BRA, RBA, RRA (the list of the 21 Oct draft, slide 222). Source:
    pathset-params.yml, MODE_WORDS; plan step 3.
    """
    if rule not in MODE_WORD_RULES:
        raise ValueError(f"MODE_WORDS must be one of {MODE_WORD_RULES}")
    air = [i for i, m in enumerate(word) if m == "A"]
    block = len(air) <= 2 and (not air or air[-1] - air[0] + 1 == len(air))
    if rule == "deck35":
        return block and word != "RAR"
    if rule == "oct21":
        return block and word not in OCT21_OUT
    return block


def class_bits(word: str) -> int:
    """Mode class of a sequence as a bit set (A = 1, B = 2, R = 4)."""
    return sum(1 << C.MODE_CODE[m] for m in set(word))


def word_tables(params: dict):
    """(allowed, prefix, wcls) over ``WORDS``: allowed under MODE_WORDS with at
    most MAX_TRANSFERS + 1 links; prefix: some allowed word starts with it;
    wcls: the mode class (``class_bits``)."""
    n_max = int(params["MAX_TRANSFERS"]) + 1
    if not 1 <= n_max <= C.MAX_LINKS:
        raise ValueError(f"MAX_TRANSFERS must lie in 0..{C.MAX_LINKS - 1}")
    allowed = np.array([len(w) <= n_max and word_allowed(w, params["MODE_WORDS"])
                        for w in WORDS])
    prefix = np.array([any(ok and v.startswith(w) for ok, v in zip(allowed, WORDS))
                       for w in WORDS])
    return allowed, prefix, np.array([class_bits(w) for w in WORDS], dtype=np.int64)


def window_tables(params: dict):
    """Transfer windows (lo, hi) in minutes indexed by the two mode codes."""
    lo = np.zeros((len(S.MODES), len(S.MODES)), dtype=np.int64)
    hi = lo.copy()
    for mi in S.MODES:
        for mo in S.MODES:
            w = params["WINDOWS"][S.window_key(mi, mo)]
            lo[C.MODE_CODE[mi], C.MODE_CODE[mo]] = w[0]
            hi[C.MODE_CODE[mi], C.MODE_CODE[mo]] = w[1]
    return lo, hi


# --- Network -------------------------------------------------------------------
class Arrays(NamedTuple):
    """The network as the compiled search reads it."""
    city: np.ndarray      # int64 (H) city of every hub
    tok: np.ndarray       # int64 (H) route-pattern token of every hub
    out_ptr: np.ndarray   # int64 (H + 1) links leaving hub h: out_link[out_ptr[h]:out_ptr[h + 1]]
    out_link: np.ndarray  # int32
    src: np.ndarray       # int32 (L)
    trg: np.ndarray       # int32 (L)
    mode: np.ndarray      # int8 (L)
    ride: np.ndarray      # float32 (L)
    freq: np.ndarray      # float32 (L)
    fmax: np.ndarray      # float32 (L) freq_max
    active: np.ndarray    # int32 (L) active_dates
    t_ptr: np.ndarray     # int64 (L + 1) ps-transfers rows with link_in l
    t_out: np.ndarray     # int32 link_out
    t_move: np.ndarray    # float32
    t_basis: np.ndarray   # uint8 BASIS_D | BASIS_T
    t_ndays: np.ndarray   # int32
    t_freq: np.ndarray    # float32
    t_ft: np.ndarray      # float32 ft_mean
    e_ptr: np.ndarray     # int64 (H + 1) egress rows of hub h
    e_dest: np.ndarray    # int32 destination as position in Net.fuas
    e_min: np.ndarray     # float32 egr_min
    fua: np.ndarray       # int64 (ND) = Net.fuas


class Net(NamedTuple):
    """Everything a shard needs; picklable (plain arrays)."""
    arrays: Arrays
    inst: TR.Instances    # dated instances of the links
    typ: TR.Instances     # typical instances
    cal: np.ndarray       # uint64 (L, N_WORDS) link calendars
    trip: np.ndarray      # int32 trip key of every event
    fuas: np.ndarray      # int64 (ND) FUA ids, ascending: connectors and car
    acc_ptr: np.ndarray   # int64 (ND + 1) access rows of the FUA at position i
    acc_hub: np.ndarray   # int32
    acc_min: np.ndarray   # float32
    car: np.ndarray       # float32 (ND, ND) car minutes; NaN without a row
    labels: np.ndarray    # object: text of every token (station: FUA id, airport: IATA)
    win_lo: np.ndarray    # int64 (3, 3) window lower bounds by mode codes
    win_hi: np.ndarray    # int64 (3, 3)


def fua_universe(connectors: pd.DataFrame, car: pd.DataFrame) -> np.ndarray:
    """FUA ids that can be an origin or a destination: those of ps-connectors
    and ps-car, ascending."""
    ids = np.concatenate([connectors["fua"].to_numpy(np.int64),
                          car["o"].to_numpy(np.int64), car["d"].to_numpy(np.int64)])
    return np.unique(ids)


def hub_cities(hubs: pd.DataFrame, connectors: pd.DataFrame) -> np.ndarray:
    """City of every hub (rows of ``hubs`` in hub order).

    Station hub: ps-hubs.fua. Airport: the fua of its ps-connectors row with
    the least acc_min, ties to the lower fua; without such a row the airport
    is its own city, -1 - hub (task definition; plan, Definitions, "Hub": an
    airport's city is the FUA with its shortest access).
    """
    hub = hubs["hub"].to_numpy(np.int64)
    city = hubs["fua"].to_numpy(np.int64).copy()
    air = (hubs["kind"] == "A").to_numpy()
    acc = connectors[np.isfinite(connectors["acc_min"].to_numpy(np.float64))]
    near = (acc.sort_values(["hub", "acc_min", "fua"], kind="stable")
            .drop_duplicates("hub").set_index("hub")["fua"])
    has = np.isin(hub, near.index.to_numpy())
    city[air] = -1 - hub[air]
    pick = air & has
    city[pick] = near.loc[hub[pick]].to_numpy(np.int64)
    return city


def _csr(keys: np.ndarray, size: int) -> np.ndarray:
    """Pointer array of rows sorted by ``keys`` in 0..size-1."""
    ptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=size), out=ptr[1:])
    return ptr


def build_network(hubs: pd.DataFrame, links: pd.DataFrame, events: pd.DataFrame,
                  masks: np.ndarray, transfers: pd.DataFrame,
                  connectors: pd.DataFrame, car: pd.DataFrame,
                  params: dict | None = None) -> Net:
    """The network of the search from the tables of the contract.

    Pure: reads only its arguments. Rows of hubs, links, transfers,
    connectors and car may come in any order (they are sorted here); events
    and masks must be the ps-events / ps-event-masks pair that ps-links.ev0,
    ev1 point into. Of the parameters, WINDOWS is read here (it belongs to the
    network: ps-transfers was counted with it); TYPICAL_SHARE must equal the
    value of ``pathset.io.P``, which ``transfer.build_typical`` reads.
    """
    params = io.P if params is None else params
    if params["TYPICAL_SHARE"] != io.P["TYPICAL_SHARE"]:
        raise ValueError("transfer.build_typical reads TYPICAL_SHARE from "
                         "pathset.io.P; build the network with that value")
    hubs = hubs.sort_values("hub", kind="stable").reset_index(drop=True)
    links = links.sort_values("link", kind="stable").reset_index(drop=True)
    n_hubs, n_links = len(hubs), len(links)
    if not (hubs["hub"].to_numpy() == np.arange(n_hubs)).all():
        raise ValueError("ps-hubs.hub must be 0 .. H-1")
    if not (links["link"].to_numpy() == np.arange(n_links)).all():
        raise ValueError("ps-links.link must be 0 .. L-1")
    conn = (connectors[["fua", "hub", "acc_min", "egr_min"]]
            .sort_values(["fua", "hub"], kind="stable").reset_index(drop=True))
    if conn.duplicated(["fua", "hub"]).any():
        raise ValueError("ps-connectors has more than one row per (fua, hub)")
    car = car[["o", "d", "minutes"]].sort_values(["o", "d"], kind="stable")
    if car.duplicated(["o", "d"]).any():
        raise ValueError("ps-car has more than one row per (o, d)")
    fuas = fua_universe(conn, car)

    # hubs: city and route-pattern token
    city = hub_cities(hubs, conn)
    kind = hubs["kind"].to_numpy()
    iata = hubs["iata"].fillna("").astype(str).to_numpy()
    label = np.where(kind == "S", hubs["fua"].astype(str).to_numpy(),
                     np.where(iata != "", iata, "A" + hubs["hub"].astype(str).to_numpy()))
    tok, labels = pd.factorize(pd.Series(label, dtype=object))

    # links by board hub
    src = links["src"].to_numpy(np.int32)
    out_link = np.argsort(src, kind="stable").astype(np.int32)

    # ps-transfers by link_in, then link_out
    t = transfers.sort_values(["link_in", "link_out"], kind="stable")
    if t.duplicated(["link_in", "link_out"]).any():
        raise ValueError("ps-transfers has more than one row per (link_in, link_out)")
    t_basis = np.where(t["basis"].to_numpy() == "D", BASIS_D, BASIS_T).astype(np.uint8)

    # egress by hub, access by FUA
    eg = conn[np.isfinite(conn["egr_min"].to_numpy(np.float64))].sort_values(
        ["hub", "fua"], kind="stable")
    ac = conn[np.isfinite(conn["acc_min"].to_numpy(np.float64))]
    car_m = np.full((len(fuas), len(fuas)), np.nan, dtype=np.float32)
    car_m[np.searchsorted(fuas, car["o"].to_numpy()),
          np.searchsorted(fuas, car["d"].to_numpy())] = car["minutes"].to_numpy(np.float32)

    arrays = Arrays(
        city=city, tok=tok.astype(np.int64),
        out_ptr=_csr(src, n_hubs), out_link=out_link,
        src=src, trg=links["trg"].to_numpy(np.int32),
        mode=np.array([C.MODE_CODE[m] for m in links["mode"]], dtype=np.int8),
        ride=links["ride"].to_numpy(np.float32),
        freq=links["freq"].to_numpy(np.float32),
        fmax=links["freq_max"].to_numpy(np.float32),
        active=links["active_dates"].to_numpy(np.int32),
        t_ptr=_csr(t["link_in"].to_numpy(np.int64), n_links),
        t_out=t["link_out"].to_numpy(np.int32),
        t_move=t["move"].to_numpy(np.float32), t_basis=t_basis,
        t_ndays=t["n_days"].to_numpy(np.int32),
        t_freq=t["freq"].to_numpy(np.float32),
        t_ft=t["ft_mean"].to_numpy(np.float32),
        e_ptr=_csr(eg["hub"].to_numpy(np.int64), n_hubs),
        e_dest=np.searchsorted(fuas, eg["fua"].to_numpy()).astype(np.int32),
        e_min=eg["egr_min"].to_numpy(np.float32),
        fua=fuas)

    # instances of the links for triple_stats
    ev = (events["dep"].to_numpy(), events["ride"].to_numpy(),
          np.ascontiguousarray(masks, dtype=np.uint64),
          links["ev0"].to_numpy(np.int32), links["ev1"].to_numpy(np.int32),
          links["tz_shift"].to_numpy(np.int32))
    win_lo, win_hi = window_tables(params)
    return Net(
        arrays=arrays, inst=TR.build_instances(*ev), typ=TR.build_typical(*ev),
        cal=TR.link_calendars(ev[2], ev[3], ev[4]),
        trip=events["trip"].to_numpy(np.int32), fuas=fuas,
        acc_ptr=_csr(np.searchsorted(fuas, ac["fua"].to_numpy()), len(fuas)),
        acc_hub=ac["hub"].to_numpy(np.int32), acc_min=ac["acc_min"].to_numpy(np.float32),
        car=car_m, labels=np.asarray(labels, dtype=object),
        win_lo=win_lo, win_hi=win_hi)


INPUTS = ("ps-hubs", "ps-links", "ps-events", "ps-transfers", "ps-connectors", "ps-car")


def load_network(params: dict | None = None) -> Net:
    """``build_network`` on the tables of PATHSET_DIR."""
    params = io.P if params is None else params
    return build_network(
        io.load_out("ps-hubs", ["hub", "kind", "fua", "iata"]),
        io.load_out("ps-links", ["link", "src", "trg", "mode", "ride", "tz_shift",
                                 "ev0", "ev1", "active_dates", "freq", "freq_max"]),
        io.load_out("ps-events", ["dep", "ride", "trip"]),
        io.load_npy("ps-event-masks"),
        io.load_out("ps-transfers", ["link_in", "link_out", "move", "basis",
                                     "n_days", "freq", "ft_mean"]),
        io.load_out("ps-connectors", ["fua", "hub", "acc_min", "egr_min"]),
        io.load_out("ps-car", ["o", "d", "minutes"]),
        params)


# --- Compiled search -----------------------------------------------------------
class Fixed(NamedTuple):
    """Constants of one shard, as the kernels read them."""
    par: np.ndarray       # cost.param_array
    prm: np.ndarray       # float64: T_MAX_MIN, MIN_ACTIVE_DATES, MIN_COMMON_DATES,
                          # MAX_TRANSFERS, MEM_CAND_GB in bytes
    win_lo: np.ndarray
    win_hi: np.ndarray
    allowed: np.ndarray   # bool (N_WORDS)
    prefix: np.ndarray    # bool (N_WORDS)
    wcls: np.ndarray      # int64 (N_WORDS)


class State(NamedTuple):
    """What the walks of one shard read and write."""
    min_gtt: np.ndarray   # float64 (ND, N_WORDS) least gtt, as float32
    best: np.ndarray      # int64 (ND, N_WORDS, 5) l1, l2, l3, q1, q2 of it
    best_tv: np.ndarray   # float64 (ND, N_WORDS, TV_COLS) its triple values
    min_gtc: np.ndarray   # float64 (ND, N_CLASSES) least gtc, as float32
    min_T: np.ndarray     # float64 (ND) least T, as float32
    mg_gtt: np.ndarray    # float64 (ND, N_CLASSES) R3 margin of gtt
    mg_gtc: np.ndarray    # float64 (ND, N_CLASSES)
    n_r0: np.ndarray      # int64 (ND, N_WORDS)
    n_r3: np.ndarray      # int64 (ND, N_WORDS)
    buf: List             # chunks int32 (CHUNK, ID_COLS): kept identities
    tvs: List             # chunks float64 (CHUNK, TV_COLS): their triple values
    stats: np.ndarray     # int64 (N_STATS)


@njit
def _srch_r32(x):
    """x as the output tables store it (float32), back in float64."""
    return np.float64(np.float32(x))


@njit
def _srch_exists(basis, n_days, min_active):
    """A chain exists when its days with an opportunity reach MIN_ACTIVE_DATES
    on basis 'D', or one weekday on basis 'T' (contract, ps-transfers)."""
    if basis == BASIS_D:
        return n_days >= min_active
    return n_days >= 1


@njit
def _srch_triple(a, inst, typ, cal, trip, work, cache, fx, st, l1, l2, l3,
                 q1, q2, m1, m2, m3):
    """(basis, n_days, counted freq, ft_mean1, ft_mean2) of a three-link chain
    from transfer.triple_stats, on typical days when a transfer joins air and
    ground; kept in ``cache``. The cache is emptied first when one more entry
    (CACHE_BYTES) would bring it and the buffer of the walk over MEM_CAND_GB
    (module docstring)."""
    n_links = a.src.shape[0]
    key = (np.int64(l1) * n_links + l2) * n_links + l3
    if key in cache:
        st.stats[ST_HITS] += 1
        return cache[key]
    forced = ((m1 == AIR) != (m2 == AIR)) or ((m2 == AIR) != (m3 == AIR))
    out = TR.triple_stats(inst, typ, cal, trip, np.int64(l1), np.int64(l2),
                          np.int64(l3), np.float64(a.t_move[q1]), fx.win_lo[m1, m2],
                          fx.win_hi[m1, m2], np.float64(a.t_move[q2]),
                          fx.win_lo[m2, m3], fx.win_hi[m2, m3],
                          np.int64(fx.prm[P_COMMON]), forced, work)
    val = (np.float64(out[0]), np.float64(out[1]), np.float64(out[2]),
           np.float64(out[3]), np.float64(out[5]))
    held = (len(cache) + 1) * CACHE_BYTES + st.stats[ST_FILL] * ID_BYTES \
        + st.stats[ST_TV] * TV_BYTES
    if len(cache) > 0 and held > fx.prm[P_BUDGET]:
        cache.clear()
        st.stats[ST_CLEAR] += 1
    cache[key] = val
    st.stats[ST_CALLS] += 1
    return val


@njit
def _srch_slot(chunks, k):
    """(chunk, row) of row k of a chunked buffer; adds a chunk of the size of
    the first one when k is past the last."""
    size = chunks[0].shape[0]
    c = k // size
    if c == len(chunks):
        chunks.append(np.empty_like(chunks[0]))
    return chunks[c], k - c * size


@njit
def _srch_need(mode, st, d, w, c, low):
    """Whether a chain with lower bounds ``low`` (T, gtt, hwt, gtc) can matter
    at destination d: pass 1, it can lower or tie the least gtt of (d, word w)
    or lower the least gtc of (d, class c) or the least T of d; pass 2, it can
    lie inside the R3 margin of (d, c)."""
    g = _srch_r32(low[1])
    h = _srch_r32(low[3])
    if mode == MIN:
        return (g <= st.min_gtt[d, w] or h < st.min_gtc[d, c]
                or _srch_r32(low[0]) < st.min_T[d])
    return g <= st.mg_gtt[d, c] or h <= st.mg_gtc[d, c]


@njit
def _srch_take(mode, st, d, w, c, T, gtt, gtc, l1, l2, l3, q1, q2, v, tvi, t_max):
    """One candidate with exact values: an R0 candidate when T <= T_MAX_MIN.
    Pass 1 updates the minima (best of (d, w): least gtt, then lower l1, l2,
    l3; with its triple values ``v``); pass 2 counts it and, when R3 keeps
    it, counts it again and writes its identity to the buffer, with the
    triple values of a three-link chain at their first kept destination
    (``tvi``: their buffer row, -1 before). Returns tvi."""
    if not T <= t_max:
        return tvi
    st.stats[ST_CAND] += 1
    g = _srch_r32(gtt)
    h = _srch_r32(gtc)
    if mode == MIN:
        b = st.best[d, w]
        if g < st.min_gtt[d, w] or (g == st.min_gtt[d, w] and (
                l1 < b[0] or (l1 == b[0] and (l2 < b[1] or (l2 == b[1] and l3 < b[2]))))):
            st.min_gtt[d, w] = g
            b[0], b[1], b[2], b[3], b[4] = l1, l2, l3, q1, q2
            for j in range(TV_COLS):
                st.best_tv[d, w, j] = v[j]
        if h < st.min_gtc[d, c]:
            st.min_gtc[d, c] = h
        t = _srch_r32(T)
        if t < st.min_T[d]:
            st.min_T[d] = t
        return tvi
    st.n_r0[d, w] += 1
    if not (g <= st.mg_gtt[d, c] or h <= st.mg_gtc[d, c]):
        return tvi
    st.n_r3[d, w] += 1
    if l3 >= 0 and tvi < 0:
        tvi = st.stats[ST_TV]
        chunk, r = _srch_slot(st.tvs, tvi)
        for j in range(TV_COLS):
            chunk[r, j] = v[j]
        st.stats[ST_TV] = tvi + 1
    k = st.stats[ST_FILL]
    chunk, r = _srch_slot(st.buf, k)
    chunk[r, 0], chunk[r, 1], chunk[r, 2], chunk[r, 3] = d, l1, l2, l3
    chunk[r, 4], chunk[r, 5], chunk[r, 6] = q1, q2, tvi
    st.stats[ST_FILL] = k + 1
    return tvi


@njit
def _srch_ends(mode, a, inst, typ, cal, trip, work, cache, st, fx, want, o_fua,
               n, acc, cf, t1a, t1b, t2a, t2b, hl, l1, l2, l3, q1, q2,
               m1, m2, m3, w):
    """All candidates of one chain: its destinations at the last hub hl.

    Clauses of R0 that depend on the end of the chain: c_last differs from
    c_first, o and the transfer cities; d differs from c_first and from the
    transfer cities (o != d: ``want`` excludes o). A chain of one or two
    links has exact values; a three-link chain is evaluated with triple_stats
    only when its bounds can matter for one of its destinations
    (``_srch_need``); otherwise pass 2 counts it by its bounds (module
    docstring, "Counting convention"; counter ST_R0B).
    """
    cl = a.city[hl]
    if (cl == cf or cl == o_fua or cl == t1a or cl == t1b or cl == t2a
            or cl == t2b):
        return
    par = fx.par
    t_max = fx.prm[P_TMAX]
    c = fx.wcls[w]
    r1 = np.float64(a.ride[l1])
    f1 = np.float64(a.fmax[l1])
    r2 = 0.0
    r3 = 0.0
    f2 = 0.0
    f3 = 0.0
    g1 = 0.0
    g2 = 0.0
    ft1 = 0.0
    ft2 = 0.0
    counted = np.float64(a.freq[l1])
    v = (np.nan, np.nan, np.nan, np.nan, np.nan)
    if n >= 2:
        r2 = np.float64(a.ride[l2])
        f2 = np.float64(a.fmax[l2])
        g1 = fx.win_lo[m1, m2] + np.float64(a.t_move[q1])
        ft1 = np.float64(a.t_ft[q1])
        counted = np.float64(a.t_freq[q1])
    s0 = a.e_ptr[hl]
    s1 = a.e_ptr[hl + 1]
    if n == 3:
        r3 = np.float64(a.ride[l3])
        f3 = np.float64(a.fmax[l3])
        g2 = fx.win_lo[m2, m3] + np.float64(a.t_move[q2])
        need = False
        for e in range(s0, s1):
            d = a.e_dest[e]
            if not want[d]:
                continue
            df = a.fua[d]
            if df == cf or df == t1a or df == t1b or df == t2a or df == t2b:
                continue
            low = C.k_bounds(3, acc, m1, m2, m3, r1, r2, r3, g1, g2,
                             np.float64(a.e_min[e]), f1, f2, f3, par)
            if low[0] <= t_max and _srch_need(mode, st, d, w, c, low):
                need = True
                break
        if not need:
            st.stats[ST_SKIP] += 1
            if mode == COLLECT:
                for e in range(s0, s1):
                    d = a.e_dest[e]
                    if not want[d]:
                        continue
                    df = a.fua[d]
                    if df == cf or df == t1a or df == t1b or df == t2a or df == t2b:
                        continue
                    low = C.k_bounds(3, acc, m1, m2, m3, r1, r2, r3, g1, g2,
                                     np.float64(a.e_min[e]), f1, f2, f3, par)
                    if low[0] <= t_max:
                        st.n_r0[d, w] += 1
                        st.stats[ST_R0B] += 1
            return
        v = _srch_triple(a, inst, typ, cal, trip, work, cache, fx, st,
                         l1, l2, l3, q1, q2, m1, m2, m3)
        if not _srch_exists(v[0], v[1], fx.prm[P_ACTIVE]):
            return
        counted = v[2]
        ft1 = v[3]
        ft2 = v[4]
    freq = C.k_capped_frequency(n, counted, f1, f2, f3)
    tvi = np.int64(-1)
    for e in range(s0, s1):
        d = a.e_dest[e]
        if not want[d]:
            continue
        df = a.fua[d]
        if df == cf or df == t1a or df == t1b or df == t2a or df == t2b:
            continue
        T, gtt, hwt, gtc = C.k_costs(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2,
                                     np.float64(a.e_min[e]), freq, par)
        tvi = _srch_take(mode, st, d, w, c, T, gtt, gtc, l1, l2, l3, q1, q2, v,
                         tvi, t_max)


@njit
def _srch_walk(mode, a, inst, typ, cal, trip, work, cache, st, fx, want, serves,
               o_fua, h0s, accs):
    """Every chain of the origin: first link from an access hub, then rows of
    ps-transfers. Prefix clauses of R0 are tested as soon as their hubs are
    known: hubs pairwise distinct; the cities of a transfer are neither o nor
    c_first; the cities of transfer 2 are not those of transfer 1; some
    allowed word begins with the modes so far. ``serves[h]``: hub h has an
    egress row to a wanted destination."""
    max_tr = np.int64(fx.prm[P_MAXTR])
    for k in range(h0s.shape[0]):
        h0 = h0s[k]
        acc = accs[k]
        cf = a.city[h0]
        for p in range(a.out_ptr[h0], a.out_ptr[h0 + 1]):
            l1 = np.int64(a.out_link[p])
            h1 = a.trg[l1]
            m1 = np.int64(a.mode[l1])
            if h1 == h0 or not fx.prefix[m1]:
                continue
            st.stats[ST_CH1] += 1
            if fx.allowed[m1] and serves[h1]:
                _srch_ends(mode, a, inst, typ, cal, trip, work, cache, st, fx, want,
                           o_fua, 1, acc, cf, NO_CITY, NO_CITY, NO_CITY, NO_CITY,
                           h1, l1, -1, -1, -1, -1, m1, 0, 0, m1)
            c1 = a.city[h1]
            if max_tr < 1 or c1 == o_fua or c1 == cf:
                continue
            for q1 in range(a.t_ptr[l1], a.t_ptr[l1 + 1]):
                l2 = np.int64(a.t_out[q1])
                h1b = a.src[l2]
                h2 = a.trg[l2]
                if h1b == h0 or h2 == h0 or h2 == h1 or h2 == h1b:
                    continue
                c1b = a.city[h1b]
                m2 = np.int64(a.mode[l2])
                w2 = 3 + 3 * m1 + m2
                if c1b == o_fua or c1b == cf or not fx.prefix[w2]:
                    continue
                st.stats[ST_CH2] += 1
                if (fx.allowed[w2] and serves[h2]
                        and _srch_exists(a.t_basis[q1], a.t_ndays[q1], fx.prm[P_ACTIVE])):
                    _srch_ends(mode, a, inst, typ, cal, trip, work, cache, st, fx,
                               want, o_fua, 2, acc, cf, c1, c1b, NO_CITY, NO_CITY,
                               h2, l1, l2, -1, q1, -1, m1, m2, 0, w2)
                c2 = a.city[h2]
                if max_tr < 2 or c2 == o_fua or c2 == cf or c2 == c1 or c2 == c1b:
                    continue
                for q2 in range(a.t_ptr[l2], a.t_ptr[l2 + 1]):
                    l3 = np.int64(a.t_out[q2])
                    h2b = a.src[l3]
                    h3 = a.trg[l3]
                    if (h2b == h0 or h2b == h1 or h2b == h1b or h3 == h0 or h3 == h1
                            or h3 == h1b or h3 == h2 or h3 == h2b or not serves[h3]):
                        continue
                    c2b = a.city[h2b]
                    m3 = np.int64(a.mode[l3])
                    w3 = 12 + 9 * m1 + 3 * m2 + m3
                    if (c2b == o_fua or c2b == cf or c2b == c1 or c2b == c1b
                            or not fx.allowed[w3]):
                        continue
                    st.stats[ST_CH3] += 1
                    _srch_ends(mode, a, inst, typ, cal, trip, work, cache, st, fx,
                               want, o_fua, 3, acc, cf, c1, c1b, c2, c2b, h3,
                               l1, l2, l3, q1, q2, m1, m2, m3, w3)


@njit
def _srch_fill(a, fx, ids, tv, acc_of, shift):
    """Exact rows of the candidates ``ids`` (d, l1, l2, l3, q1, q2, row of
    ``tv``): the float columns ``FLOATS`` and the integers n (transfers),
    word, h_first, h_last, basis, n_days and the route-pattern tokens
    (consecutive repeats removed, -1 after the last). A three-link chain
    takes basis, n_days, counted freq, ft1 and ft2 from its row of ``tv``
    (triple_stats values of the walk). Each row is checked against its lower
    bounds (cost.k_bounds_hold, bounds raised by ``shift``); returns the
    position of the first violation, or -1."""
    rows = ids.shape[0]
    F = np.full((rows, N_F), np.nan, dtype=np.float32)
    I = np.full((rows, N_I), -1, dtype=np.int32)
    hubs = np.empty(N_TOK, dtype=np.int64)
    par = fx.par
    for i in range(rows):
        d = ids[i, 0]
        l1, l2, l3 = np.int64(ids[i, 1]), np.int64(ids[i, 2]), np.int64(ids[i, 3])
        q1, q2 = np.int64(ids[i, 4]), np.int64(ids[i, 5])
        n = 1 if l2 < 0 else (2 if l3 < 0 else 3)
        m1 = np.int64(a.mode[l1])
        m2 = 0
        m3 = 0
        r1 = np.float64(a.ride[l1])
        f1 = np.float64(a.fmax[l1])
        r2 = 0.0
        r3 = 0.0
        f2 = 0.0
        f3 = 0.0
        g1 = 0.0
        g2 = 0.0
        mv1 = np.nan
        mv2 = np.nan
        ft1 = np.nan
        ft2 = np.nan
        counted = np.float64(a.freq[l1])
        basis = np.int64(BASIS_D)
        n_days = np.int64(a.active[l1])
        w = m1
        hubs[0] = a.src[l1]
        hubs[1] = a.trg[l1]
        k = 2
        if n >= 2:
            m2 = np.int64(a.mode[l2])
            r2 = np.float64(a.ride[l2])
            f2 = np.float64(a.fmax[l2])
            mv1 = np.float64(a.t_move[q1])
            g1 = fx.win_lo[m1, m2] + mv1
            w = 3 + 3 * m1 + m2
            if a.src[l2] != hubs[k - 1]:
                hubs[k] = a.src[l2]
                k += 1
            hubs[k] = a.trg[l2]
            k += 1
        if n == 2:
            counted = np.float64(a.t_freq[q1])
            basis = np.int64(a.t_basis[q1])
            n_days = np.int64(a.t_ndays[q1])
            ft1 = np.float64(a.t_ft[q1])
        if n == 3:
            m3 = np.int64(a.mode[l3])
            r3 = np.float64(a.ride[l3])
            f3 = np.float64(a.fmax[l3])
            mv2 = np.float64(a.t_move[q2])
            g2 = fx.win_lo[m2, m3] + mv2
            w = 12 + 9 * m1 + 3 * m2 + m3
            if a.src[l3] != hubs[k - 1]:
                hubs[k] = a.src[l3]
                k += 1
            hubs[k] = a.trg[l3]
            k += 1
            v = tv[ids[i, 6]]
            basis = np.int64(v[0])
            n_days = np.int64(v[1])
            counted = v[2]
            ft1 = v[3]
            ft2 = v[4]
        hl = hubs[k - 1]
        egr = np.nan
        for e in range(a.e_ptr[hl], a.e_ptr[hl + 1]):
            if a.e_dest[e] == d:
                egr = np.float64(a.e_min[e])
                break
        acc = acc_of[hubs[0]]
        freq = C.k_capped_frequency(n, counted, f1, f2, f3)
        comp = C.k_components(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par)
        true = C.k_costs(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, freq, par)
        low = C.k_bounds(n, acc, m1, m2, m3, r1, r2, r3, g1, g2, egr, f1, f2, f3, par)
        if not C.k_bounds_hold((low[0] + shift, low[1] + shift, low[2] + shift,
                                low[3] + shift), true):
            return F, I, i
        vals = (acc, egr, mv1, mv2, ft1, ft2, comp[0], comp[1], comp[2], comp[3],
                comp[4], comp[5], comp[6], true[0], true[1], true[2],
                C.k_hidden_wait(np.float64(a.freq[l1]), par), true[3], freq, counted)
        for j in range(N_F):
            F[i, j] = vals[j]
        I[i, 0] = n - 1
        I[i, 1] = w
        I[i, 2] = hubs[0]
        I[i, 3] = hl
        I[i, 4] = basis
        I[i, 5] = n_days
        last = -1
        j = 0
        for t in range(k):
            tk = a.tok[hubs[t]]
            if tk != last:
                I[i, 6 + j] = tk
                j += 1
                last = tk
    return F, I, -1


# --- One shard -----------------------------------------------------------------
def _patterns(tokens: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Route patterns: token rows joined with '>' (one Python join per
    distinct row)."""
    if len(tokens) == 0:
        return np.array([], dtype=object)
    uniq, inv = np.unique(tokens, axis=0, return_inverse=True)
    text = np.array([">".join(labels[t] for t in row if t >= 0) for row in uniq.tolist()],
                    dtype=object)
    return text[inv.ravel()]


def _keys(chains: np.ndarray) -> np.ndarray:
    """Link ids '-'-joined (one Python join per distinct chain)."""
    if len(chains) == 0:
        return np.array([], dtype=object)
    uniq, inv = np.unique(chains, axis=0, return_inverse=True)
    text = np.array(["-".join(str(x) for x in row if x >= 0) for row in uniq.tolist()],
                    dtype=object)
    return text[inv.ravel()]


def class_minima(per_word: np.ndarray, wcls: np.ndarray) -> np.ndarray:
    """Minimum over the words of each mode class, per destination."""
    out = np.full((per_word.shape[0], N_CLASSES), np.inf)
    for j in range(N_WORDS):
        out[:, wcls[j]] = np.minimum(out[:, wcls[j]], per_word[:, j])
    return out


def _chunks(dtype, cols: int) -> List:
    """A chunked buffer (numba typed List) with one empty chunk of CHUNK rows."""
    out = List()
    out.append(np.empty((CHUNK, cols), dtype=dtype))
    return out


class Shard:
    """Search state of one origin FUA: kernel arguments, the triple_stats
    cache, the buffer of the last pass 2 and the counters of the walks."""

    def __init__(self, net: Net, o: int, params: dict):
        self.net, self.params, self.o = net, params, int(o)
        fuas = net.fuas
        i = int(np.searchsorted(fuas, self.o))
        if i >= len(fuas) or fuas[i] != self.o:
            raise ValueError(f"FUA {o} has no ps-connectors or ps-car row")
        win_lo, win_hi = window_tables(params)
        if not (np.array_equal(win_lo, net.win_lo) and np.array_equal(win_hi, net.win_hi)):
            raise ValueError("WINDOWS differ from those the network was built "
                             "with (ps-transfers was counted with them)")
        self.oi = i
        s, e = net.acc_ptr[i], net.acc_ptr[i + 1]
        self.h0s = net.acc_hub[s:e].astype(np.int64)
        self.accs = net.acc_min[s:e].astype(np.float64)
        self.acc_of = np.full(len(net.arrays.city), np.nan)
        self.acc_of[self.h0s] = self.accs
        allowed, prefix, wcls = word_tables(params)
        self.budget = float(params["MEM_CAND_GB"]) * 2 ** 30
        prm = np.array([params["T_MAX_MIN"], params["MIN_ACTIVE_DATES"],
                        params["MIN_COMMON_DATES"], params["MAX_TRANSFERS"],
                        self.budget], dtype=np.float64)
        self.fx = Fixed(C.param_array(params), prm, net.win_lo, net.win_hi,
                        allowed, prefix, wcls)
        self.cache = Dict.empty(key_type=types.int64,
                                value_type=types.UniTuple(types.float64, 5))
        self.work = TR.workspace(net.inst, net.typ)
        nd = len(fuas)
        self.st = State(
            min_gtt=np.full((nd, N_WORDS), np.inf),
            best=np.full((nd, N_WORDS, 5), -1, dtype=np.int64),
            best_tv=np.full((nd, N_WORDS, TV_COLS), np.nan),
            min_gtc=np.full((nd, N_CLASSES), np.inf), min_T=np.full(nd, np.inf),
            mg_gtt=np.full((nd, N_CLASSES), np.inf),
            mg_gtc=np.full((nd, N_CLASSES), np.inf),
            n_r0=np.zeros((nd, N_WORDS), dtype=np.int64),
            n_r3=np.zeros((nd, N_WORDS), dtype=np.int64),
            buf=_chunks(np.int32, ID_COLS), tvs=_chunks(np.float64, TV_COLS),
            stats=np.zeros(N_STATS, dtype=np.int64))
        self.n_ids = 0          # rows in the buffer
        self.n_tv = 0
        self.log: dict = {}     # counters and seconds per pass
        self.tag = ""           # prefix of the pass names in the log

    def want(self, dests=None) -> np.ndarray:
        """Destination mask over Net.fuas: all FUAs, or the FUA ids ``dests``
        that are known; never the origin."""
        if dests is None:
            want = np.ones(len(self.net.fuas), dtype=bool)
        else:
            want = np.isin(self.net.fuas, np.asarray(list(dests), dtype=np.int64))
        want[self.oi] = False
        return want

    def _walk(self, mode: int, want: np.ndarray, name: str) -> None:
        net, a = self.net, self.net.arrays
        rows = want[a.e_dest]
        serves = np.zeros(len(a.city), dtype=bool)
        serves[np.repeat(np.arange(len(a.city)), np.diff(a.e_ptr))[rows]] = True
        self.st.stats[:] = 0
        start = time.perf_counter()
        _srch_walk(mode, a, net.inst, net.typ, net.cal, net.trip, self.work,
                   self.cache, self.st, self.fx, want, serves, a.fua[self.oi],
                   self.h0s, self.accs)
        rec = self.log.setdefault(self.tag + name, {"s": 0.0, "cache_entries": 0,
                                                    **{k: 0 for k in STAT_NAMES}})
        rec["s"] = round(rec["s"] + time.perf_counter() - start, 3)
        rec["cache_entries"] = max(rec["cache_entries"], len(self.cache))
        for k, v in zip(STAT_NAMES, self.st.stats.tolist()):
            rec[k] += v

    def minima(self, want: np.ndarray) -> None:
        """Pass 1: least gtt per (d, word) with its candidate and its triple
        values, least gtc per (d, class), least T per d, over the R0
        candidates (exact)."""
        st = self.st
        for arr in (st.min_gtt, st.min_gtc, st.min_T):
            arr[...] = np.inf
        st.best[...] = -1
        st.best_tv[...] = np.nan
        self._walk(MIN, want, "pass1")

    def margins(self, mu: float) -> None:
        """R3 margins (1 + MU) x the class minima of pass 1 (inf: no cut)."""
        st = self.st
        if math.isinf(mu):
            st.mg_gtt[...] = np.inf
            st.mg_gtc[...] = np.inf
        else:
            st.mg_gtt[...] = (1 + mu) * class_minima(st.min_gtt, self.fx.wcls)
            st.mg_gtc[...] = (1 + mu) * st.min_gtc

    def collect(self, want: np.ndarray) -> None:
        """Pass 2: R0 candidates and those R3 keeps, per (d, word), and the
        buffer of the kept ones (identities and triple values)."""
        self.st.n_r0[...] = 0
        self.st.n_r3[...] = 0
        self.free_buffer()
        self._walk(COLLECT, want, "pass2")
        self.n_ids = int(self.st.stats[ST_FILL])
        self.n_tv = int(self.st.stats[ST_TV])

    def free_buffer(self) -> None:
        self.st = self.st._replace(buf=_chunks(np.int32, ID_COLS),
                                   tvs=_chunks(np.float64, TV_COLS))
        self.n_ids = self.n_tv = 0

    def release_cache(self) -> None:
        """Empty the triple_stats cache (its values are in the buffer)."""
        self.cache.clear()

    def buffer_bytes(self) -> int:
        """Bytes allocated by the buffer of the last pass 2."""
        st = self.st
        return sum(c.nbytes for c in st.buf) + sum(c.nbytes for c in st.tvs)

    def kept_ids(self, mask: np.ndarray) -> np.ndarray:
        """Buffer rows of the destinations in ``mask``, in the order of the
        walk."""
        parts, left = [], self.n_ids
        for chunk in self.st.buf:
            if left <= 0:
                break
            rows = chunk[:left]
            left -= len(rows)
            parts.append(rows[mask[rows[:, 0]]])
        return np.concatenate(parts) if parts else np.empty((0, ID_COLS), dtype=np.int32)

    def triple_values(self) -> np.ndarray:
        """Triple values of the buffer, one row per kept three-link chain."""
        return np.concatenate(list(self.st.tvs))[:self.n_tv]

    def rows(self, ids: np.ndarray, tv: np.ndarray) -> pd.DataFrame:
        """PATH rows of the candidates ``ids`` (buffer rows; column 6 the row
        of ``tv`` of a three-link chain), exact and checked against their
        bounds."""
        net = self.net
        F, I, bad = _srch_fill(net.arrays, self.fx,
                               np.ascontiguousarray(ids, dtype=np.int32),
                               np.ascontiguousarray(tv, dtype=np.float64).reshape(-1, TV_COLS),
                               self.acc_of, float(BOUND_TEST_SHIFT))
        if bad >= 0:
            raise BoundViolation(
                f"origin {self.o}: a lower bound exceeds the exact value of the "
                f"candidate (d, l1, l2, l3) = "
                f"({net.fuas[ids[bad, 0]]}, {ids[bad, 1]}, {ids[bad, 2]}, {ids[bad, 3]})")
        cols = {"o": self.o, "d": net.fuas[ids[:, 0]],
                "seq": np.array(WORDS, dtype=object)[I[:, 1]], "n": I[:, 0],
                "h_first": I[:, 2], "h_last": I[:, 3],
                "l1": ids[:, 1], "l2": ids[:, 2], "l3": ids[:, 3],
                "cities": _patterns(I[:, 6:], net.labels), "key": _keys(ids[:, 1:4]),
                **{name: F[:, j] for j, name in enumerate(FLOATS)},
                "basis": np.where(I[:, 4] == BASIS_D, "D", "T"), "n_days": I[:, 5]}
        return as_table(pd.DataFrame(cols), "path")

    def car_rows(self, want: np.ndarray) -> pd.DataFrame:
        """The direct car alternative of the wanted pairs (INCLUDE_CAR)."""
        if not self.params["INCLUDE_CAR"]:
            return empty_table("path")
        minutes = self.net.car[self.oi]
        d = np.flatnonzero(want & np.isfinite(minutes))
        m = minutes[d].astype(np.float64)
        cost = C.car_costs(m, self.fx.par)
        nan = np.full(len(d), np.nan)
        cols = {"o": self.o, "d": self.net.fuas[d], "seq": "C", "n": 0,
                "h_first": -1, "h_last": -1, "l1": -1, "l2": -1, "l3": -1,
                "cities": "", "key": "car", "acc": 0.0, "egr": 0.0, "mv1": nan,
                "mv2": nan, "ft1": nan, "ft2": nan,
                **{k: cost[k] for k in C.COMPONENTS + ("T", "gtt", "hwt", "gtc")},
                "hwt_first": 0.0, "freq": np.inf, "freq_counted": np.inf,
                "basis": "", "n_days": 0}
        return as_table(pd.DataFrame(cols, index=np.arange(len(d))), "path")

    def candidates(self, want: np.ndarray, mu: float) -> pd.DataFrame:
        """PATH rows of the R0 candidates of the wanted destinations that R3
        at ``mu`` keeps (all of them at mu = inf), the car included; sorted by
        (d, l1, l2, l3)."""
        if not math.isinf(mu):
            self.minima(want)
        self.margins(mu)
        self.collect(want)
        self.release_cache()
        out = _concat([self.rows(self.kept_ids(want), self.triple_values()),
                       self.car_rows(want)], "path")
        self.free_buffer()
        return out.sort_values(["d", "l1", "l2", "l3"], kind="stable").reset_index(drop=True)


# --- Rules per pair ------------------------------------------------------------
def decide(cands: pd.DataFrame, params: dict):
    """rules.apply_rules on the candidates of ONE FUA pair (PATH rows).

    Candidate ids are positions in increasing (l1, l2, l3), the car (l1 = -1)
    first: a subset keeps the relative order of the full set, so the R3-kept
    candidates and all R0 candidates are decided alike. Returns (cands with
    the column cand, alternatives, trace) as apply_rules gives them.
    """
    c = cands.sort_values(["l1", "l2", "l3"], kind="stable").reset_index(drop=True)
    c["cand"] = np.arange(len(c), dtype=np.int64)
    inp = pd.DataFrame({"cand": c["cand"], "seq": c["seq"], "n": c["n"],
                        "T": c["T"], "gtt": c["gtt"], "gtc": c["gtc"],
                        "freq": c["freq"], "first_link": c["l1"], "key": c["key"],
                        "cities": c["cities"]})
    alts, trace = rules.apply_rules(inp, params, guarantee=bool(params["R4_GUARANTEE"]))
    return c, alts, trace


def pair_tables(c: pd.DataFrame, alts: pd.DataFrame, params: dict):
    """Rows of ps-alternatives and ps-variants of one pair from ``decide``.

    Alternatives: the representative's PATH row, rank 1.. in increasing gtc
    (the order of apply_rules). Variants: the members other than the
    representative in increasing (T, gtt, key), at most VARIANTS_KEPT per
    alternative, numbered 1.. (``member``).
    """
    reps = alts["alt"].to_numpy(np.int64)
    out = c.iloc[reps][S.PATH].reset_index(drop=True)
    out["rank"] = np.arange(1, len(alts) + 1)
    for k in ("cls", "freq_pooled", "n_variants", "status"):
        out[k] = alts[k].to_numpy()
    idx, rank, member = [], [], []
    for r, (alt, members) in enumerate(zip(reps.tolist(), alts["members"]), 1):
        others = [m for m in members if m != alt][:int(params["VARIANTS_KEPT"])]
        idx += others
        rank += [r] * len(others)
        member += range(1, len(others) + 1)
    var = c.iloc[idx][S.PATH].reset_index(drop=True)
    var["rank"] = rank
    var["member"] = member
    return out, var


def trace_table(c: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """ps-trace rows of one pair: every candidate with its fate."""
    if not (trace["cand"].to_numpy() == c["cand"].to_numpy()).all():
        raise RuntimeError("trace and candidates are not aligned")
    out = c[S.PATH + ["cand"]].copy()
    for k in ("cls", "alt", "status", "rule", "by", "detail"):
        out[k] = trace[k].to_numpy()
    return out


def unserved(funnel: pd.DataFrame) -> pd.DataFrame:
    """ps-unserved: pairs of the funnel without a service alternative; reason
    'nothing' when there is no car alternative either."""
    f = funnel[~funnel["pt"].to_numpy(bool)]
    reason = np.where(np.isfinite(f["car_min"].to_numpy(np.float64)),
                      "no_service_path", "nothing")
    return as_table(pd.DataFrame({"o": f["o"].to_numpy(), "d": f["d"].to_numpy(),
                                  "reason": reason}), "unserved")


def dest_blocks(kept: np.ndarray, want: np.ndarray, limit_bytes: float) -> list:
    """Destinations (positions) in blocks whose kept candidates take at most
    ``limit_bytes`` at CAND_BYTES each; a destination above the limit is a
    block of its own. Plan step 3: never truncate."""
    blocks, cur, size = [], [], 0.0
    for d in np.flatnonzero(want).tolist():
        b = float(kept[d]) * CAND_BYTES
        if cur and size + b > limit_bytes:
            blocks.append(cur)
            cur, size = [], 0.0
        cur.append(d)
        size += b
    if cur:
        blocks.append(cur)
    return blocks


def _nan_inf(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nan)


# --- Public API ----------------------------------------------------------------
def candidates(net: Net, o: int, dests, params: dict | None = None,
               mu: float = math.inf) -> pd.DataFrame:
    """PATH rows of every R0 candidate of the pairs (o, d), d in ``dests``
    (FUA ids), with the car; at a finite ``mu`` only those R3 keeps. Sorted by
    (d, l1, l2, l3). For the viewer and the tests."""
    params = io.P if params is None else params
    sh = Shard(net, o, params)
    return sh.candidates(sh.want(dests), float(mu))


def minima(net: Net, o: int, params: dict | None = None):
    """Pass 1 on its own: (per_word, per_class). per_word has o, d, seq,
    min_gtt and the best candidate's l1, l2, l3 for every (d, word) with an
    R0 candidate; per_class has o, d, cls (letters), min_gtc."""
    params = io.P if params is None else params
    sh = Shard(net, o, params)
    want = sh.want()
    sh.minima(want)
    st = sh.st
    d, w = np.nonzero(np.isfinite(st.min_gtt) & want[:, None])
    per_word = pd.DataFrame({"o": sh.o, "d": net.fuas[d], "seq": np.array(WORDS)[w],
                             "min_gtt": st.min_gtt[d, w], "l1": st.best[d, w, 0],
                             "l2": st.best[d, w, 1], "l3": st.best[d, w, 2]})
    d, c = np.nonzero(np.isfinite(st.min_gtc) & want[:, None])
    names = np.array(["".join(m for m in S.MODES if b & (1 << C.MODE_CODE[m]))
                      for b in range(N_CLASSES)])
    per_class = pd.DataFrame({"o": sh.o, "d": net.fuas[d], "cls": names[c],
                              "min_gtc": st.min_gtc[d, c]})
    return per_word, per_class


def run_origin(net: Net, o: int, params: dict | None = None,
               named_dests=()) -> dict:
    """Alternatives of every pair (o, d): the tables of one shard.

    Pass 1 (minima), pass 2 (counts and the buffer of the kept candidates),
    the cache emptied, then by destination block (MEM_CAND_GB less the
    buffer) the exact rows and per pair rules.apply_rules on the kept
    candidates and the car (R4_GUARANTEE as its ``guarantee``); WINDOWS must
    be those of the network. Returns a dict of DataFrames in contract columns and
    dtypes: alternatives, variants, best, funnel, seq_counts and trace (pairs
    (o, d) with d in ``named_dests``: every R0 candidate, enumerated with
    MU = inf, with its fate under apply_rules at the parameters' MU). The
    funnel has one row per FUA d != o of the network; its attrs["search"]
    holds the counters and seconds of the passes.

    Funnel (car included): n_r0 R0 candidates (module docstring, "Counting
    convention"), n_r3 kept by R3, n_r1 alternatives entering R2, n_r2 after
    R2 and the comparison across classes, n_capped dropped by R5, n_kept
    alternatives, n_classes their mode classes, best_T and best_gtt the least
    T and gtt of the R0 service candidates, car_min the car minutes (NaN
    without a car alternative), pt: a service alternative is kept.
    """
    params = io.P if params is None else params
    rules.check_params(params)
    started = time.perf_counter()
    sh = Shard(net, o, params)
    st, fuas = sh.st, net.fuas
    want = sh.want()
    sh.minima(want)
    sh.margins(float(params["MU"]))
    sh.collect(want)
    sh.release_cache()
    n_r0, n_r3 = st.n_r0.copy(), st.n_r3.copy()
    kept = n_r3.sum(axis=1)
    car_min = np.where(want, net.car[sh.oi], np.nan) if params["INCLUDE_CAR"] \
        else np.full(len(fuas), np.nan)
    has_car = np.isfinite(car_min)

    alts, variants, counts = [], [], {}
    buffer_mb = round(sh.buffer_bytes() / 2 ** 20, 1)
    blocks = dest_blocks(kept, want, sh.budget - sh.buffer_bytes())
    tv = sh.triple_values()
    rules_s = 0.0
    for block in blocks:
        mask = np.zeros(len(fuas), dtype=bool)
        mask[block] = True
        cands = _concat([sh.rows(sh.kept_ids(mask), tv), sh.car_rows(mask)], "path")
        t = time.perf_counter()
        for d, grp in cands.groupby("d", sort=True):
            c, a, tr = decide(grp, params)
            if (tr["status"] == "cut_r3").any():
                raise RuntimeError(f"pair ({o}, {d}): rules.apply_rules cut "
                                   "candidates that R3 of the search kept")
            A, V = pair_tables(c, a, params)
            alts.append(A)
            variants.append(V)
            n_capped = int((tr["status"] == "capped").sum())
            counts[int(d)] = (int(tr.loc[tr["alt"] >= 0, "alt"].nunique()),
                              len(a) + n_capped, len(a), n_capped,
                              int(a["cls"].nunique()), bool((a["seq"] != "C").any()))
        rules_s += time.perf_counter() - t
    alternatives = _concat(alts, "alternatives")
    del tv
    sh.free_buffer()

    # ps-best-by-sequence: the candidates of pass 1, and the car
    d_pos, w = np.nonzero(np.isfinite(st.min_gtt) & want[:, None])
    best_ids = np.column_stack([d_pos, st.best[d_pos, w],
                                np.arange(len(d_pos))]).astype(np.int32)
    best = _concat([sh.rows(best_ids, st.best_tv[d_pos, w]), sh.car_rows(want)], "best")
    best = best.sort_values(["d", "seq"], kind="stable").reset_index(drop=True)

    # funnel and counts per sequence
    dd = np.flatnonzero(want)
    stats = np.array([counts.get(int(fuas[d]), (0, 0, 0, 0, 0, False)) for d in dd],
                     dtype=object).reshape(len(dd), 6)
    funnel = as_table(pd.DataFrame({
        "o": sh.o, "d": fuas[dd], "n_r0": n_r0[dd].sum(axis=1) + has_car[dd],
        "n_r3": kept[dd] + has_car[dd], "n_r1": stats[:, 0].astype(np.int64),
        "n_r2": stats[:, 1].astype(np.int64), "n_kept": stats[:, 2].astype(np.int64),
        "n_capped": stats[:, 3].astype(np.int64), "n_classes": stats[:, 4].astype(np.int64),
        "best_T": _nan_inf(st.min_T[dd]), "best_gtt": _nan_inf(st.min_gtt[dd].min(axis=1)),
        "car_min": car_min[dd], "pt": stats[:, 5].astype(bool)}), "funnel")
    n_kept = alternatives.groupby("seq").size()
    words = [w for w, ok in zip(WORDS, sh.fx.allowed) if ok]
    rows = [(w, int(n_r0[want, WORD_OF[w]].sum()), int(n_r3[want, WORD_OF[w]].sum()))
            for w in words]
    if params["INCLUDE_CAR"]:
        rows.append(("C", int(has_car.sum()), int(has_car.sum())))
    seq_counts = as_table(pd.DataFrame({
        "o": sh.o, "seq": [r[0] for r in rows], "n_r0": [r[1] for r in rows],
        "n_r3": [r[2] for r in rows],
        "n_kept": [int(n_kept.get(r[0], 0)) for r in rows]}), "seq_counts")

    # trace of the named pairs: every R0 candidate (MU = inf) with its fate
    traces = []
    named = sh.want([d for d in named_dests if int(d) != sh.o])
    if named.any():
        sh.tag = "trace_"
        full = sh.candidates(named, math.inf)
        for _, grp in full.groupby("d", sort=True):
            c, _, tr = decide(grp, params)
            traces.append(trace_table(c, tr))
    trace = _concat(traces, "trace")

    funnel.attrs["search"] = {
        "origin": sh.o, "blocks": len(blocks), "buffer_mb": buffer_mb,
        "rules_s": round(rules_s, 3),
        "total_s": round(time.perf_counter() - started, 3), **sh.log}
    return {"alternatives": alternatives, "variants": _concat(variants, "variants"),
            "best": best, "funnel": funnel, "seq_counts": seq_counts, "trace": trace}


# --- Several origins, several processes ----------------------------------------
def named_dests_of(o: int, params: dict) -> list:
    """Destinations of the NAMED_PAIRS whose origin is o."""
    return [int(d) for a, d in params["NAMED_PAIRS"] if int(a) == int(o)]


def _worker(tasks, results, params, named, net):
    """Process body: build the network once, then run origins from the queue
    until a None; the first error is reported and ends the worker."""
    try:
        net = load_network(params) if net is None else net
    except BaseException:
        results.put((None, None, traceback.format_exc()))
        return
    while True:
        o = tasks.get()
        if o is None:
            return
        try:
            out = run_origin(net, o, params, named_dests_of(o, params) if named else ())
        except BaseException:
            results.put((o, None, traceback.format_exc()))
            return
        results.put((o, out, None))


def run_pool(origins, params: dict | None = None, workers: int = 1,
             named: bool = False, net: Net | None = None):
    """Yield (o, tables of ``run_origin``) for every origin, in completion order.

    One worker runs in this process; more are spawned processes, each with
    NUMBA_NUM_THREADS = 1, that build the network once (``load_network``, or
    the pickled ``net`` when one is given) and take origins from a queue. The
    first error of any origin raises RuntimeError here, with the worker's
    traceback, and stops the other workers. ``named``: trace the NAMED_PAIRS
    of each origin.
    """
    params = io.P if params is None else params
    origins = [int(o) for o in origins]
    if workers <= 1 or len(origins) <= 1:
        net = load_network(params) if net is None else net
        for o in origins:
            yield o, run_origin(net, o, params,
                                named_dests_of(o, params) if named else ())
        return
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    n_proc = min(int(workers), len(origins))
    for o in origins:
        tasks.put(o)
    for _ in range(n_proc):
        tasks.put(None)
    procs = [ctx.Process(target=_worker, args=(tasks, results, params, named, net),
                         daemon=True) for _ in range(n_proc)]
    saved = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        for p in procs:
            p.start()
    finally:
        if saved is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved
    try:
        left = len(origins)
        while left:
            try:
                o, out, err = results.get(timeout=2.0)
            except queue.Empty:
                dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"a worker ended with exit code {dead[0]}")
                continue
            if err is not None:
                raise RuntimeError(f"origin {o} failed in a worker:\n{err}")
            left -= 1
            yield o, out
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join()
