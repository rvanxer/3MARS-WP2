"""Opportunities of a chain of two or three service links.

Rules (plan, section Definitions; contract in ``pathset/schemas.py``):

Transfer      from an arrival at clock a to a departure at clock d of the next
              link, feasible when  lo + move <= d - a <= hi  and the two runs
              do not have the same trip key (the same train continuing is not
              a transfer).
Opportunity   a feasible itinerary of the chain such that no other feasible
              itinerary leaves no earlier and arrives no later; itineraries
              with the same first departure and last arrival count once.
f_trans       mean arrival-to-departure minutes over the opportunities.
Frequency     median, over the days of first departure that have an
              opportunity, of the opportunities that day.
Basis         'D', date by date, when the links share at least
              MIN_COMMON_DATES active dates; otherwise 'T', typical days: per
              weekday the events that run on at least TYPICAL_SHARE of the
              link's active dates of that weekday within one IATA season, the
              season whose week best matches the link's own (``_typical_flags``),
              on one periodic week.

Layout. The events of a link are rows [ev0, ev1) of ``dep`` (local minute at
the board hub), ``ride``, ``trip`` and ``masks`` (calendar of departure dates,
``pathset.dates``). An INSTANCE is one event on one date. ``Instances`` holds
the instances of all links in CSR form, sorted per link by (departure, arrival,
event row): ``dep`` and ``arr`` are absolute minutes, day * 1440 + minute, and
``arr`` is the clock at the alight hub (dep + ride + tz_shift). The typical
instances use the same structure with the minute of the week, weekday * 1440 +
minute, Monday = 0.

One of several itineraries with the same first departure and last arrival
stands for the opportunity: it is chosen from the last link backwards, each
time the latest instance in the sort order. Its transfer times enter f_trans.

The chain kernels write into a WORKSPACE, an int32 array (ROWS, capacity) made
once per thread by ``workspace``: allocating inside a parallel loop costs page
faults that serialise the threads. A workspace that is too small for a call is
replaced for that call only, so its size affects speed, never results. Two
threads must never share one.

The jitted functions take every threshold as an argument; only the wrappers
``typical_flags``, ``build_typical`` and ``transfer_table`` read
``pathset.io.P``. No timeline constant is compiled in, so the numba cache
stays valid when the parameter file changes.

Consumers
---------
pathset-transfers.py, from Python:

    inst = build_instances(dep, ride, masks, ev0, ev1, tz_shift)
    typ  = build_typical(dep, ride, masks, ev0, ev1, tz_shift)
    cal  = link_calendars(masks, ev0, ev1)
    out  = transfer_table(inst, typ, cal, trip, link_in, link_out, move, lo,
                          hi, typical)     # typical: air <-> ground pairs

with the ps-events columns dep, ride, trip, ps-event-masks.npy, the ps-links
columns ev0, ev1, tz_shift, and per pair the move (minutes) and the window of
``schemas.window_key``. ``out`` holds numpy arrays of the ps-transfers columns
(basis as a code: ``basis_labels``).

The search, from inside its own @njit loops (no Python object is needed;
Instances is a NamedTuple of arrays that numba passes as a tuple):

    work = workspace(inst, typ)            # once per thread, outside the loops
    basis, n_days, freq, ft_mean1, ft_min1, ft_mean2, ft_min2, elapsed = \
        triple_stats(inst, typ, cal, trip, a, b, c, move1, lo1, hi1,
                     move2, lo2, hi2, min_common, typical, work)
    basis, n_days, freq, ft_mean, ft_min, elapsed = \
        pair_stats(inst, typ, cal, trip, a, b, move, lo, hi, min_common,
                   typical, work)

    a, b, c        int, link ids (rows of ps-links)
    move*          float, minutes between the hubs of the transfer (0 in one)
    lo*, hi*       int, the transfer window in minutes (WINDOWS)
    min_common     int, MIN_COMMON_DATES read once by the caller
    typical        bool, True to count on typical days whatever the common
                   dates (plan: always when the chain mixes air and ground)
    basis          int, BASIS_D (ord 'D') or BASIS_T (ord 'T')
    n_days         int, dates (D) or weekdays, at most 7 (T), of first
                   departure with at least one opportunity; 0 = no chain
    freq           float, median over those days of the opportunities that
                   day (departures per day); 0.0 without an opportunity
    ft_mean*, ft_min*  float, minutes from arrival to departure at the
                   first and the second transfer, mean (f_trans) and smallest
                   over the opportunities; NaN without an opportunity
    elapsed        float, mean minutes from first departure to last arrival
                   over the opportunities, transfers included; NaN without one

One call of triple_stats or pair_stats is sequential; parallel callers give
every thread its own ``work``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
from numba import njit, prange

from . import dates as D
from .io import P, log

DAY = 1440
WEEK = 7 * DAY
NONE = -(2 ** 31)        # "no chain ends here" in arrays of first departures
ALL = 2 ** 31 - 1        # horizon that keeps every first departure
BASIS_D = ord("D")
BASIS_T = ord("T")
CHUNK = 1 << 20          # pairs per call of the parallel kernel
BLOCK = 256              # pairs a thread takes at a time

# Rows of the workspace: best feeder and first departure per instance of the
# second and of the last link, sort index, the opportunities (first departure,
# last arrival, minutes of each transfer), day counts and their histogram, and
# the typical week of up to three links laid out over several weeks.
(BEST1, FIRST1, BEST2, FIRST2, IDX, O_FIRST, O_LAST, O_FT1, O_FT2, COUNTS,
 HIST, WEEKS_A, WEEKS_B, WEEKS_C) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 17)
ROWS = 20
WEEKS_ROOM = 4           # weeks of a typical link a workspace holds


class Instances(NamedTuple):
    """Instances of all links, link l in rows [ptr[l], ptr[l + 1])."""
    ptr: np.ndarray      # int64 (L + 1)
    dep: np.ndarray      # int32 departure, absolute minute or minute of week
    arr: np.ndarray      # int32 arrival clock at the alight hub, same origin
    ev: np.ndarray       # int32 event row
    dmin: np.ndarray     # int32 (L) smallest ride + tz_shift of the link
    dmax: np.ndarray     # int32 (L) largest ride + tz_shift of the link


class Link(NamedTuple):
    """Instances of one link, sorted by (dep, arr, ev)."""
    dep: np.ndarray
    arr: np.ndarray
    ev: np.ndarray
    dmin: int
    dmax: int


# --- Calendars and typical days ----------------------------------------------
def weekday_masks() -> np.ndarray:
    """Calendar masks (7, N_WORDS) of the Mondays ... Sundays of the timeline."""
    wd = D.weekdays()
    return D.pack_bool(np.stack([wd == w for w in range(7)]))


@njit(cache=True, parallel=True)
def link_calendars(masks, ev0, ev1):
    """Active dates of every link: the OR of the calendars of its events
    (plan, Definitions, "Active date")."""
    cal = np.zeros((ev0.shape[0], masks.shape[1]), dtype=np.uint64)
    for l in prange(ev0.shape[0]):
        for e in range(ev0[l], ev1[l]):
            D.or_into(cal[l], masks[e])
    return cal


def season_masks() -> np.ndarray:
    """Calendar masks (S, N_WORDS) of the scheduling seasons of the timeline:
    maximal runs of days with the same EU summer-time flag
    (``dates.eu_summer_days``), which are the IATA summer and winter seasons.
    Airlines retime their flights from one season to the next."""
    summer = D.eu_summer_days()
    season = np.concatenate([[0], np.cumsum(summer[1:] != summer[:-1])])
    return D.pack_bool(np.stack([season == s for s in range(season[-1] + 1)]))


def _weekday_of_bit() -> np.ndarray:
    """Weekday of every bit of a calendar mask; -1 past the timeline."""
    wd = np.full(D.N_WORDS * 64, -1, dtype=np.int8)
    wd[:D.N_DAYS] = D.weekdays()
    return wd


@njit(cache=True)
def _and3_count(a, b, c):
    """Number of days three calendars share."""
    n = 0
    for w in range(a.shape[0]):
        n += D.popcount64(a[w] & b[w] & c[w])
    return n


@njit(cache=True)
def _link_week(masks, e0, e1, wd, counts, buf, hist):
    """The link's own week: the sum over weekdays w of the median, over the
    link's active dates of weekday w, of its departures that date (the rule
    of ps-links.freq, per weekday). ``counts`` and ``buf`` have one entry
    per mask bit, ``hist`` e1 - e0 + 1."""
    counts[:] = 0
    for e in range(e0, e1):
        for w in range(masks.shape[1]):
            x = masks[e, w]
            if x == 0:
                continue
            for b in range(64):
                if (x >> np.uint64(b)) & np.uint64(1):
                    counts[w * 64 + b] += 1
    total = 0.0
    for w in range(7):
        n = 0
        for d in range(counts.shape[0]):
            if counts[d] > 0 and wd[d] == w:
                buf[n] = counts[d]
                n += 1
        if n > 0:
            total += _median(buf, n, hist)
    return total


@njit(cache=True)
def _event_weekdays(mask, season, wd_masks, active, share):
    """Weekdays (bit w) on which an event runs on at least ``share`` of the
    link's active dates of weekday w within the season (``active[w]``)."""
    bits = 0
    for w in range(7):
        runs = _and3_count(mask, season, wd_masks[w])
        if runs > 0 and runs + 1e-9 >= share * active[w]:
            bits |= 1 << w
    return bits


@njit(cache=True, parallel=True)
def _typical_flags(masks, ev0, ev1, wd_masks, seasons, wd, share):
    """Bit w of flags[e] is set when event e is typical on weekday w.

    Rule (plan, Definitions, "Typical days", scoped to one season after the
    review of 13 Nov): within a season, the departures typical on weekday w
    are the events that run on at least ``share`` of the link's active dates
    of weekday w in that season. The link's typical week is the season whose
    typical departures over the week come closest in number to the link's
    own week (``_link_week``); on a tie the season with more active dates,
    then the later one. Over the whole timeline no retimed version of a
    flight runs on half of a weekday's dates, so a share test over all dates
    keeps none of them.
    """
    flags = np.zeros(masks.shape[0], dtype=np.uint8)
    for l in prange(ev0.shape[0]):
        e0, e1 = ev0[l], ev1[l]
        if e1 == e0:
            continue
        cal = np.zeros(masks.shape[1], dtype=np.uint64)
        for e in range(e0, e1):
            D.or_into(cal, masks[e])
        week = _link_week(masks, e0, e1, wd, np.empty(wd.shape[0], np.int32),
                          np.empty(wd.shape[0], np.int32),
                          np.empty(e1 - e0 + 1, np.int32))
        active = np.zeros(7, dtype=np.int64)
        best, best_gap, best_act = -1, np.inf, -1
        for s in range(seasons.shape[0]):
            act = D.and_count(cal, seasons[s])
            if act == 0:
                continue
            for w in range(7):
                active[w] = _and3_count(cal, seasons[s], wd_masks[w])
            n = 0
            for e in range(e0, e1):
                n += D.popcount64(_event_weekdays(masks[e], seasons[s],
                                                  wd_masks, active, share))
            gap = abs(n - week)
            if gap < best_gap or (gap == best_gap and act >= best_act):
                best, best_gap, best_act = s, gap, act
        for w in range(7):
            active[w] = _and3_count(cal, seasons[best], wd_masks[w])
        for e in range(e0, e1):
            flags[e] = np.uint8(_event_weekdays(masks[e], seasons[best],
                                                wd_masks, active, share))
    return flags


def typical_flags(masks, ev0, ev1) -> np.ndarray:
    """Typical weekdays of every event as a bit set (bit 0 = Monday), at the
    share TYPICAL_SHARE, within the season chosen for its link
    (``_typical_flags``)."""
    return _typical_flags(np.ascontiguousarray(masks, dtype=np.uint64),
                          _i32(ev0), _i32(ev1), weekday_masks(), season_masks(),
                          _weekday_of_bit(), float(P["TYPICAL_SHARE"]))


# --- Instances ---------------------------------------------------------------
@njit(cache=True)
def _order(dep, ride, e0, e1):
    """Offsets from e0 of the rows e0..e1-1 in (dep, ride, row) order."""
    key = np.empty(e1 - e0, dtype=np.int64)
    for q in range(e1 - e0):
        key[q] = np.int64(dep[e0 + q]) * 65536 + np.int64(ride[e0 + q])
    return np.argsort(key, kind="mergesort")


@njit(cache=True, parallel=True)
def _count_dated(masks, ev0, ev1):
    n = np.zeros(ev0.shape[0], dtype=np.int64)
    for l in prange(ev0.shape[0]):
        c = 0
        for e in range(ev0[l], ev1[l]):
            c += D.popcount_row(masks[e])
        n[l] = c
    return n


@njit(cache=True, parallel=True)
def _count_typical(flags, ev0, ev1):
    n = np.zeros(ev0.shape[0], dtype=np.int64)
    for l in prange(ev0.shape[0]):
        c = 0
        for e in range(ev0[l], ev1[l]):
            c += D.popcount64(flags[e])
        n[l] = c
    return n


@njit(cache=True, parallel=True)
def _fill_dated(dep, ride, masks, ev0, ev1, tz_shift, ptr, o_dep, o_arr, o_ev):
    """Instances of every link, day by day and within a day in event order,
    which is the (dep, arr, ev) order because 0 <= dep < 1440."""
    one = np.uint64(1)
    for l in prange(ev0.shape[0]):
        e0 = ev0[l]
        order = _order(dep, ride, e0, ev1[l])
        q = ptr[l]
        for d in range(masks.shape[1] * 64):
            w = d >> 6
            b = np.uint64(d & 63)
            for o in order:
                e = e0 + o
                if (masks[e, w] >> b) & one:
                    t = d * DAY + dep[e]
                    o_dep[q] = t
                    o_arr[q] = t + ride[e] + tz_shift[l]
                    o_ev[q] = e
                    q += 1


@njit(cache=True, parallel=True)
def _fill_typical(dep, ride, flags, ev0, ev1, tz_shift, ptr, o_dep, o_arr, o_ev):
    """Typical instances of every link on one week, in (dep, arr, ev) order."""
    for l in prange(ev0.shape[0]):
        e0 = ev0[l]
        order = _order(dep, ride, e0, ev1[l])
        q = ptr[l]
        for w in range(7):
            for o in order:
                e = e0 + o
                if (flags[e] >> w) & 1:
                    t = w * DAY + dep[e]
                    o_dep[q] = t
                    o_arr[q] = t + ride[e] + tz_shift[l]
                    o_ev[q] = e
                    q += 1


@njit(cache=True)
def _durations(ride, ev0, ev1, tz_shift):
    """Smallest and largest ride + tz_shift of every link (0 without events)."""
    dmin = np.zeros(ev0.shape[0], dtype=np.int32)
    dmax = np.zeros(ev0.shape[0], dtype=np.int32)
    for l in range(ev0.shape[0]):
        if ev1[l] > ev0[l]:
            dmin[l] = ride[ev0[l]:ev1[l]].min() + tz_shift[l]
            dmax[l] = ride[ev0[l]:ev1[l]].max() + tz_shift[l]
    return dmin, dmax


def _i32(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.int32)


def _events(dep, ride, ev0, ev1, tz_shift):
    dep, ride = _i32(dep), _i32(ride)
    if dep.size and (dep.min() < 0 or dep.max() >= DAY):
        raise ValueError("dep must be a local clock minute, 0..1439")
    return dep, ride, _i32(ev0), _i32(ev1), _i32(tz_shift)


def _assemble(n, fill, dep, ride, marks, ev0, ev1, tz_shift) -> Instances:
    ptr = np.zeros(n.shape[0] + 1, dtype=np.int64)
    np.cumsum(n, out=ptr[1:])
    o_dep = np.empty(ptr[-1], dtype=np.int32)
    o_arr = np.empty(ptr[-1], dtype=np.int32)
    o_ev = np.empty(ptr[-1], dtype=np.int32)
    fill(dep, ride, marks, ev0, ev1, tz_shift, ptr, o_dep, o_arr, o_ev)
    dmin, dmax = _durations(ride, ev0, ev1, tz_shift)
    return Instances(ptr, o_dep, o_arr, o_ev, dmin, dmax)


def build_instances(dep, ride, masks, ev0, ev1, tz_shift) -> Instances:
    """Dated instances of all links: one per event and departure date, with
    absolute minutes day * 1440 + dep and arrival = departure + ride +
    tz_shift of the link (contract, ps-links.tz_shift)."""
    dep, ride, ev0, ev1, tz_shift = _events(dep, ride, ev0, ev1, tz_shift)
    masks = np.ascontiguousarray(masks, dtype=np.uint64)
    return _assemble(_count_dated(masks, ev0, ev1), _fill_dated,
                     dep, ride, masks, ev0, ev1, tz_shift)


def build_typical(dep, ride, masks, ev0, ev1, tz_shift) -> Instances:
    """Typical instances of all links on one periodic week (minute of week =
    weekday * 1440 + dep). Arrivals may pass the end of the week or, with a
    negative tz_shift, fall before its start: the consumer wraps."""
    dep, ride, ev0, ev1, tz_shift = _events(dep, ride, ev0, ev1, tz_shift)
    flags = typical_flags(masks, ev0, ev1)
    return _assemble(_count_typical(flags, ev0, ev1), _fill_typical,
                     dep, ride, flags, ev0, ev1, tz_shift)


def link_instances(dep, ride, masks, tz_shift: int):
    """Dated instances of one link given its events: (inst_dep, inst_arr,
    inst_ev), sorted by absolute departure; inst_ev indexes the given rows."""
    inst = build_instances(dep, ride, masks, [0], [len(dep)], [tz_shift])
    return inst.dep, inst.arr, inst.ev


def instance_bytes(n_instances: int, n_links: int) -> int:
    """Memory of an ``Instances`` structure: three int32 per instance, one
    int64 pointer and two int32 durations per link. n_instances of the dated
    structure is the sum of ps-events.n_dates."""
    return 12 * int(n_instances) + 8 * (int(n_links) + 1) + 8 * int(n_links)


@njit(cache=True)
def link_view(inst, l):
    """Instances of link l as a ``Link`` (views, nothing is copied)."""
    s = inst.ptr[l]
    e = inst.ptr[l + 1]
    return Link(inst.dep[s:e], inst.arr[s:e], inst.ev[s:e],
                np.int64(inst.dmin[l]), np.int64(inst.dmax[l]))


# --- Workspace ---------------------------------------------------------------
def workspace_size(inst: Instances, typ: Instances) -> int:
    """Capacity that holds any link of ``inst`` and WEEKS_ROOM weeks of any
    link of ``typ``."""
    dated = int(np.diff(inst.ptr).max(initial=0))
    typical = int(np.diff(typ.ptr).max(initial=0))
    return max(dated, WEEKS_ROOM * typical) + 1


def workspace(inst: Instances, typ: Instances) -> np.ndarray:
    """A workspace for ``pair_stats`` and ``triple_stats``; one per thread."""
    return np.empty((ROWS, workspace_size(inst, typ)), dtype=np.int32)


@njit(cache=True)
def _room(work, need):
    """The workspace, or a new one when it has fewer than ``need`` columns."""
    if work.shape[1] >= need:
        return work
    return np.empty((ROWS, need), dtype=np.int32)


@njit(cache=True)
def _weeks_size(typ, l, t0, t1):
    """Instances of link l in the weeks that hold the minutes t0..t1."""
    return (typ.ptr[l + 1] - typ.ptr[l]) * max(t1 // WEEK - t0 // WEEK + 1, 0)


@njit(cache=True)
def _weeks(typ, l, t0, t1, work, row):
    """Typical instances of link l repeated over the weeks that hold the
    minutes t0..t1 (week 0 = minutes 0..10079), written to three rows of the
    workspace. The copies stay sorted."""
    s = typ.ptr[l]
    n = typ.ptr[l + 1] - s
    w0 = t0 // WEEK
    k = max(t1 // WEEK - w0 + 1, 0)
    for c in range(k):
        shift = (w0 + c) * WEEK
        for i in range(n):
            work[row, c * n + i] = typ.dep[s + i] + shift
            work[row + 1, c * n + i] = typ.arr[s + i] + shift
            work[row + 2, c * n + i] = typ.ev[s + i]
    return Link(work[row, :n * k], work[row + 1, :n * k], work[row + 2, :n * k],
                np.int64(typ.dmin[l]), np.int64(typ.dmax[l]))


# --- Chain kernels -----------------------------------------------------------
@njit(cache=True)
def _feed(a, val, slack, b, trip, gmin, hi, best, first):
    """Best feeder of every instance of link b among the instances of link a.

    Instance i feeds instance j when  gmin <= dep_b[j] - arr_a[i] <= hi  and
    the trip keys differ (contract, last paragraph). ``val[i]`` is the latest
    first departure of a chain that ends with i (NONE without one). Of the
    feeders of j the one with the largest val is kept, the later instance on a
    tie: best[j] = its index or -1, first[j] = its val or NONE.

    val[i] <= dep_a[i] - slack must hold: with departures ascending this
    bounds what earlier instances can offer and ends the backward scan.
    """
    na = a.dep.shape[0]
    p = 0                                 # instances from p on arrive too late
    for j in range(b.dep.shape[0]):
        latest = b.dep[j] - gmin          # arrival bounds at the hub
        earliest = b.dep[j] - hi
        while p < na and a.dep[p] + a.dmin <= latest:
            p += 1
        bi = -1
        bv = NONE
        i = p - 1
        while i >= 0 and a.dep[i] + a.dmax >= earliest and a.dep[i] - slack > bv:
            if (val[i] > bv and earliest <= a.arr[i] <= latest
                    and trip[a.ev[i]] != trip[b.ev[j]]):
                bi = i
                bv = val[i]
            i -= 1
        best[j] = bi
        first[j] = bv


@njit(cache=True)
def _front(first, arr, idx):
    """Pareto filter (plan, Definitions, "Opportunity"). ``first[j]`` is the
    latest first departure of a chain ending with instance j of the last link
    (NONE without one) and ``arr[j]`` its arrival. Writes to idx[:n] the
    instances whose (first, arr) no other one beats or equals with a later
    departure or an earlier arrival, in ascending first departure, and returns
    n. Of equal (first, arr) the later instance is kept.

    Insertion sort by (arr ascending, first descending): the input is nearly in
    that order already because the instances are sorted by departure.
    """
    m = 0
    for j in range(arr.shape[0]):
        if first[j] == NONE:
            continue
        q = m
        while q > 0:
            p = idx[q - 1]
            if arr[p] > arr[j] or (arr[p] == arr[j] and first[p] < first[j]):
                idx[q] = p
                q -= 1
            else:
                break
        idx[q] = j
        m += 1
    n = 0
    top = NONE                            # latest first departure so far
    top_arr = 0
    for q in range(m):
        j = idx[q]
        if first[j] > top:
            top = first[j]
            top_arr = arr[j]
            idx[n] = j
            n += 1
        elif first[j] == top and arr[j] == top_arr:
            idx[n - 1] = j
    return n


@njit(cache=True)
def _chain2(a, b, trip, gmin, hi, work):
    """Opportunities of a -> b into the rows O_FIRST, O_LAST and O_FT1 of the
    workspace, in ascending first departure; returns their number. The
    workspace needs one column per instance of b."""
    best, first, idx = work[BEST2], work[FIRST2], work[IDX]
    _feed(a, a.dep, 0, b, trip, gmin, hi, best, first)
    n = _front(first, b.arr, idx)
    for q in range(n):
        j = idx[q]
        i = best[j]
        work[O_FIRST, q] = a.dep[i]
        work[O_LAST, q] = b.arr[j]
        work[O_FT1, q] = b.dep[j] - a.arr[i]
    return n


@njit(cache=True)
def _chain3(a, b, c, trip, gmin1, hi1, gmin2, hi2, work):
    """Opportunities of a -> b -> c by dynamic programming: the best feeder of
    every instance of b, then of every instance of c (the feeder whose own
    chain leaves latest), then the Pareto filter. An instance of b without an
    onward connection never reaches c, so a first departure that misses the
    only connection cannot hide one that makes it. Output as ``_chain2`` plus
    row O_FT2; the workspace needs one column per instance of b and of c."""
    best1, first1 = work[BEST1], work[FIRST1]
    best2, first2, idx = work[BEST2], work[FIRST2], work[IDX]
    _feed(a, a.dep, 0, b, trip, gmin1, hi1, best1, first1)
    # a chain through instance j of b left no later than dep_b[j] - gmin1 - dmin_a
    _feed(b, first1, gmin1 + a.dmin, c, trip, gmin2, hi2, best2, first2)
    n = _front(first2, c.arr, idx)
    for q in range(n):
        k = idx[q]
        j = best2[k]
        i = best1[j]
        work[O_FIRST, q] = a.dep[i]
        work[O_LAST, q] = c.arr[k]
        work[O_FT1, q] = b.dep[j] - a.arr[i]
        work[O_FT2, q] = c.dep[k] - b.arr[j]
    return n


@njit(cache=True)
def opportunities2(a, b, trip, gmin, hi):
    """Opportunities of the chain a -> b (two ``Link``): arrays first (first
    departure, ascending), last (last arrival) and ft (1, n) with the
    arrival-to-departure minutes of the transfer. ``gmin`` is lo + move."""
    work = np.empty((ROWS, b.dep.shape[0]), dtype=np.int32)
    n = _chain2(a, b, trip, gmin, hi, work)
    return (work[O_FIRST, :n].copy(), work[O_LAST, :n].copy(),
            work[O_FT1:O_FT1 + 1, :n].copy())


@njit(cache=True)
def opportunities3(a, b, c, trip, gmin1, hi1, gmin2, hi2):
    """Opportunities of the chain a -> b -> c: first, last and ft (2, n), as
    ``opportunities2``."""
    work = np.empty((ROWS, max(b.dep.shape[0], c.dep.shape[0])), dtype=np.int32)
    n = _chain3(a, b, c, trip, gmin1, hi1, gmin2, hi2, work)
    return (work[O_FIRST, :n].copy(), work[O_LAST, :n].copy(),
            work[O_FT1:O_FT2 + 1, :n].copy())


@njit(cache=True)
def _median(counts, n, hist):
    """Median of the positive integers counts[:n], n > 0, as np.median gives
    it (the rule of pt-links.py), by counting into ``hist``."""
    top = 0
    for q in range(n):
        top = max(top, counts[q])
    hist[:top + 1] = 0
    for q in range(n):
        hist[counts[q]] += 1
    lower = -1
    seen = 0
    for c in range(top + 1):
        seen += hist[c]
        if lower < 0 and seen > (n - 1) // 2:
            lower = c
        if seen > n // 2:
            return 0.5 * (lower + c)
    return np.nan


@njit(cache=True)
def _stats(work, n, horizon):
    """Statistics of the n opportunities in the workspace whose first
    departure lies in [0, horizon):

    n_days   days of first departure with at least one opportunity
    freq     median over those days of the opportunities that day (plan,
             Definitions, "Path frequency")
    ft_mean1, ft_min1, ft_mean2, ft_min2
             per transfer, mean (f_trans) and smallest arrival-to-departure
             minutes; the second pair is meaningless for a two-link chain
    elapsed  mean of last arrival - first departure

    Without an opportunity: 0, 0.0 and NaN. The workspace needs n + 1 columns.
    """
    first, counts = work[O_FIRST], work[COUNTS]
    n_days = 0
    kept = 0
    day = -1
    span = 0.0
    sum1 = 0.0
    sum2 = 0.0
    low1 = ALL
    low2 = ALL
    for q in range(n):
        if first[q] < 0 or first[q] >= horizon:
            continue
        if first[q] // DAY != day:
            day = first[q] // DAY
            counts[n_days] = 0
            n_days += 1
        counts[n_days - 1] += 1
        kept += 1
        span += work[O_LAST, q] - first[q]
        sum1 += work[O_FT1, q]
        sum2 += work[O_FT2, q]
        low1 = min(low1, work[O_FT1, q])
        low2 = min(low2, work[O_FT2, q])
    if kept == 0:
        return 0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan
    return (n_days, _median(counts, n_days, work[HIST]), sum1 / kept, float(low1),
            sum2 / kept, float(low2), span / kept)


@njit(cache=True)
def _onward(t0, t1, dmin, dmax, gmin, hi):
    """Departures of the next link that runs leaving in t0..t1 and lasting
    dmin..dmax can connect to, as a range of minutes."""
    return t0 + dmin + gmin, t1 + dmax + hi


@njit(cache=True)
def pair_stats(inst, typ, cal, trip, a, b, move, lo, hi, min_common, typical,
               work):
    """Transfer from link a to link b: (basis, n_days, freq, ft_mean, ft_min,
    elapsed), see ``_stats``. Date by date when the two calendars ``cal``
    share at least ``min_common`` dates and ``typical`` is False, else on
    typical days, where n_days counts weekdays (plan, Definitions, "Typical
    days").

    Typical days are one periodic week. An itinerary and one that beats it
    differ in elapsed time by at most ``spread``, so link a is laid out from
    minute 0 to the end of the week plus the spread and link b over everything
    it can reach from there; only first departures of week 0 are counted. This
    is where Sunday connects to Monday.
    """
    gmin = math.ceil(lo + move)
    if not typical and D.and_count(cal[a], cal[b]) >= min_common:
        basis = BASIS_D
        horizon = ALL
        la = link_view(inst, a)
        lb = link_view(inst, b)
        work = _room(work, lb.dep.shape[0] + 1)
    else:
        basis = BASIS_T
        horizon = WEEK
        spread = ((typ.dmax[a] + hi + typ.dmax[b])
                  - (typ.dmin[a] + gmin + typ.dmin[b]))
        t1 = WEEK - 1 + max(spread, 0)
        u0, u1 = _onward(0, t1, typ.dmin[a], typ.dmax[a], gmin, hi)
        work = _room(work, max(_weeks_size(typ, a, 0, t1),
                               _weeks_size(typ, b, u0, u1)) + 1)
        la = _weeks(typ, a, 0, t1, work, WEEKS_A)
        lb = _weeks(typ, b, u0, u1, work, WEEKS_B)
    n = _chain2(la, lb, trip, gmin, hi, work)
    n_days, freq, mean1, low1, _, _, elapsed = _stats(work, n, horizon)
    return basis, n_days, freq, mean1, low1, elapsed


@njit(cache=True)
def triple_stats(inst, typ, cal, trip, a, b, c, move1, lo1, hi1,
                 move2, lo2, hi2, min_common, typical, work):
    """Chain a -> b -> c: (basis, n_days, freq, ft_mean1, ft_min1, ft_mean2,
    ft_min2, elapsed); basis and layout as in ``pair_stats``, on the dates all
    three links share."""
    gmin1 = math.ceil(lo1 + move1)
    gmin2 = math.ceil(lo2 + move2)
    if not typical and _and3_count(cal[a], cal[b], cal[c]) >= min_common:
        basis = BASIS_D
        horizon = ALL
        la = link_view(inst, a)
        lb = link_view(inst, b)
        lc = link_view(inst, c)
        work = _room(work, max(lb.dep.shape[0], lc.dep.shape[0]) + 1)
    else:
        basis = BASIS_T
        horizon = WEEK
        spread = ((typ.dmax[a] + hi1 + typ.dmax[b] + hi2 + typ.dmax[c])
                  - (typ.dmin[a] + gmin1 + typ.dmin[b] + gmin2 + typ.dmin[c]))
        t1 = WEEK - 1 + max(spread, 0)
        u0, u1 = _onward(0, t1, typ.dmin[a], typ.dmax[a], gmin1, hi1)
        v0, v1 = _onward(u0, u1, typ.dmin[b], typ.dmax[b], gmin2, hi2)
        work = _room(work, max(_weeks_size(typ, a, 0, t1),
                               _weeks_size(typ, b, u0, u1),
                               _weeks_size(typ, c, v0, v1)) + 1)
        la = _weeks(typ, a, 0, t1, work, WEEKS_A)
        lb = _weeks(typ, b, u0, u1, work, WEEKS_B)
        lc = _weeks(typ, c, v0, v1, work, WEEKS_C)
    n = _chain3(la, lb, lc, trip, gmin1, hi1, gmin2, hi2, work)
    n_days, freq, mean1, low1, mean2, low2, elapsed = _stats(work, n, horizon)
    return basis, n_days, freq, mean1, low1, mean2, low2, elapsed


# --- Transfer table ----------------------------------------------------------
@njit(cache=True, parallel=True)
def _pairs(inst, typ, cal, trip, link_in, link_out, move, lo, hi, typical,
           min_common, works, basis, n_days, freq, ft_mean, ft_min):
    """``pair_stats`` of every pair; fills the five outputs. The parallel loop
    runs over blocks of BLOCK pairs; a thread uses its own workspace
    ``works[numba.get_thread_id()]`` (works has one per possible thread)."""
    n = link_in.shape[0]
    for k in prange((n + BLOCK - 1) // BLOCK):
        work = works[numba.get_thread_id()]
        for p in range(k * BLOCK, min((k + 1) * BLOCK, n)):
            b, days, f, mean, low, _ = pair_stats(
                inst, typ, cal, trip, link_in[p], link_out[p], move[p], lo[p],
                hi[p], min_common, typical[p], work)
            basis[p] = b
            n_days[p] = days
            freq[p] = f
            ft_mean[p] = mean
            ft_min[p] = low


def _check_pairs(inst, typ, cal, trip, link_in, link_out, move, lo, hi, typical):
    """Raise ValueError on input the kernel would turn into silent wrong rows:
    pair arrays of different lengths, link ids outside the links of ``inst``,
    structures of different links, a trip key array that does not cover the
    event rows, or lo > hi."""
    n = link_in.shape[0]
    if any(x.shape[0] != n for x in (link_out, move, lo, hi, typical)):
        raise ValueError("link_in, link_out, move, lo, hi and typical differ in length")
    n_links = inst.ptr.shape[0] - 1
    if typ.ptr.shape[0] != n_links + 1 or cal.shape[0] != n_links:
        raise ValueError(f"inst has {n_links} links; typ and cal must have as many")
    if inst.ev.shape[0] and trip.shape[0] <= int(inst.ev.max()):
        raise ValueError("trip has fewer rows than the events of inst")
    if n and (min(link_in.min(), link_out.min()) < 0
              or max(link_in.max(), link_out.max()) >= n_links):
        raise ValueError(f"link ids must lie in 0..{n_links - 1}")
    if (lo > hi).any():
        raise ValueError("a transfer window has lo > hi")


def transfer_table(inst: Instances, typ: Instances, cal, trip, link_in,
                   link_out, move, lo, hi, typical=None) -> dict:
    """Columns basis, n_days, freq, ft_mean, ft_min of ps-transfers for the
    pairs (link_in, link_out) with their move and window [lo, hi] in minutes.

    ``typical`` marks pairs counted on typical days whatever their common
    dates (plan: always for air <-> ground). basis is returned as a code,
    BASIS_D or BASIS_T (``basis_labels``). Inconsistent input raises
    ValueError (``_check_pairs``).

    Scheduling. The pairs go through the parallel kernel CHUNK at a time (one
    log line per chunk); inside it a thread takes one block of BLOCK pairs at
    a time (dynamic, ``numba.parallel_chunksize(1)``) and keeps one workspace
    for all its blocks. Rows sorted by link_in put the pairs of a busy hub
    next to each other, and the cores of the machine differ in speed, so a
    static split would leave threads idle. Measured on the network of 13 Nov
    (78,659 links, 50.3M dated instances), all 14,992,073 pairs inside one
    hub in link_in order, 22 threads, Core Ultra 9 185H shared with other
    jobs, runs differing by up to a factor 3:
      static split, a workspace per block      0.22 - 0.90 M pairs/s (7 runs)
      dynamic blocks, a workspace per block    0.29 - 0.91 M pairs/s (8 runs)
      dynamic blocks, a workspace per thread   0.48 - 0.97 M pairs/s (4 runs)
      one thread                               0.30 - 0.36 M pairs/s
    The last layout (this one) was faster than the second in 5 of 6 paired
    runs and has the highest worst case. A loop that shares nothing gains
    only 3-5 times over one thread on this machine, at 6-8 threads.
    """
    n = len(link_in)
    link_in, link_out, lo, hi = _i32(link_in), _i32(link_out), _i32(lo), _i32(hi)
    move = np.ascontiguousarray(move, dtype=np.float32)
    trip = _i32(trip)
    cal = np.ascontiguousarray(cal, dtype=np.uint64)
    typical = (np.zeros(n, dtype=np.bool_) if typical is None
               else np.ascontiguousarray(typical, dtype=np.bool_))
    _check_pairs(inst, typ, cal, trip, link_in, link_out, move, lo, hi, typical)
    basis = np.empty(n, dtype=np.uint8)
    n_days = np.empty(n, dtype=np.int32)
    freq = np.empty(n, dtype=np.float32)
    ft_mean = np.empty(n, dtype=np.float32)
    ft_min = np.empty(n, dtype=np.float32)
    min_common = int(P["MIN_COMMON_DATES"])
    works = np.empty((numba.config.NUMBA_NUM_THREADS, ROWS,
                      workspace_size(inst, typ)), dtype=np.int32)
    with numba.parallel_chunksize(1):
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            _pairs(inst, typ, cal, trip, link_in[s:e], link_out[s:e], move[s:e],
                   lo[s:e], hi[s:e], typical[s:e], min_common, works,
                   basis[s:e], n_days[s:e], freq[s:e], ft_mean[s:e], ft_min[s:e])
            if n > CHUNK:
                log(f"transfers: {e:,} of {n:,} pairs")
    return {"basis": basis, "n_days": n_days, "freq": freq,
            "ft_mean": ft_mean, "ft_min": ft_min}


def basis_labels(codes: np.ndarray) -> np.ndarray:
    """Basis codes of the kernels as the 'D' / 'T' strings of ps-transfers."""
    return np.where(np.asarray(codes) == BASIS_D, "D", "T")
