"""Transfer kernels against hand cases and a brute-force reference.

The reference (``ref_*``) is pure Python and shares no code with the kernels:
it lists every feasible itinerary of a chain, then applies the Pareto filter
of the plan's definition of an opportunity.
"""

import datetime as dt
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import dates as D  # noqa: E402
from pathset import transfer as T  # noqa: E402
from pathset.io import P  # noqa: E402

DAY, WEEK = 1440, 10080
SHARE = P["TYPICAL_SHARE"]
MIN_COMMON = P["MIN_COMMON_DATES"]
WINDOWS = [tuple(w) for w in P["WINDOWS"].values()]
MONDAY = 1                        # day index of Monday 2 January 2023
ALL_DAYS = range(D.N_DAYS)


def hm(h, m=0):
    return 60 * h + m


class Net:
    """A few links. A link is (tz_shift, [(dep, ride, trip, days), ...]); the
    events may come in any order."""

    def __init__(self, *links):
        self.links = links
        events = [e for _, evs in links for e in evs]
        self.dep = np.array([e[0] for e in events], dtype=np.int16)
        self.ride = np.array([e[1] for e in events], dtype=np.int16)
        self.trip = np.array([e[2] for e in events], dtype=np.int32)
        active = np.zeros((len(events), D.N_DAYS), dtype=bool)
        for row, e in enumerate(events):
            active[row, list(e[3])] = True
        self.masks = D.pack_bool(active)
        self.ev1 = np.cumsum([len(evs) for _, evs in links]).astype(np.int32)
        self.ev0 = (self.ev1 - [len(evs) for _, evs in links]).astype(np.int32)
        self.tz = np.array([tz for tz, _ in links], dtype=np.int16)
        args = (self.dep, self.ride, self.masks, self.ev0, self.ev1, self.tz)
        self.inst = T.build_instances(*args)
        self.typ = T.build_typical(*args)
        self.cal = T.link_calendars(self.masks, self.ev0, self.ev1)
        self.work = T.workspace(self.inst, self.typ)

    def pair(self, lo, hi, move=0.0, basis="D", a=0, b=1):
        """(n_days, freq, ft_mean, ft_min, elapsed) of a -> b on one basis."""
        out = T.pair_stats(self.inst, self.typ, self.cal, self.trip, a, b,
                           float(move), int(lo), int(hi), 0, basis == "T", self.work)
        assert chr(out[0]) == basis
        return out[1:]

    def triple(self, w1, w2, basis="D"):
        """(n_days, freq, ft_mean1, ft_min1, ft_mean2, ft_min2, elapsed) of
        link 0 -> 1 -> 2; a window is (lo, move, hi)."""
        out = T.triple_stats(
            self.inst, self.typ, self.cal, self.trip, 0, 1, 2,
            float(w1[1]), int(w1[0]), int(w1[2]),
            float(w2[1]), int(w2[0]), int(w2[2]), 0, basis == "T", self.work)
        assert chr(out[0]) == basis
        return out[1:]


def daily(*links):
    """Net of links that run every day; an event is (dep, ride, trip)."""
    return Net(*[(tz, [(*e, ALL_DAYS) for e in evs]) for tz, evs in links])


# --- Brute-force reference ---------------------------------------------------
def ref_dated(net, l):
    """Dated instances of link l as sorted (dep, arr, row)."""
    tz, evs = net.links[l]
    return sorted((d * DAY + dep, d * DAY + dep + ride + tz, int(net.ev0[l]) + q)
                  for q, (dep, ride, _, days) in enumerate(evs) for d in set(days))


def ref_seasons():
    """IATA season of every day: a new one starts on the last Sunday of March
    and on the last Sunday of October."""
    starts = {max(dt.date(y, m, d) for d in range(25, 32) if dt.date(y, m, d).weekday() == 6)
              for y in range(D.DAY0.year, D.DAY1.year + 1) for m in (3, 10)}
    out = [0]
    for d in range(1, D.N_DAYS):
        out.append(out[-1] + (D.index_date(d) in starts))
    return out


SEASON = ref_seasons()
WD = [D.index_date(d).weekday() for d in range(D.N_DAYS)]


def ref_typical_events(evs):
    """(event, weekday) pairs of the typical week: the share test within each
    season; the season whose count is closest to the link's own week (sum
    over weekdays of the median departures per active date), then the one
    with more active dates, then the later one."""
    days = [set(e[3]) for e in evs]
    active = set().union(*days)
    per_date = Counter(d for ds in days for d in ds)
    week = sum(statistics.median([per_date[d] for d in active if WD[d] == w])
               for w in range(7) if any(WD[d] == w for d in active))
    best = None
    for s in sorted({SEASON[d] for d in active}):
        act = Counter(WD[d] for d in active if SEASON[d] == s)
        typ = []
        for q, ds in enumerate(days):
            runs = Counter(WD[d] for d in ds if SEASON[d] == s)
            typ += [(q, w) for w in range(7) if runs[w] > 0 and runs[w] >= SHARE * act[w]]
        key = (abs(len(typ) - week), -sum(act.values()), -s)
        if best is None or key < best[0]:
            best = (key, typ)
    return best[1]


def ref_typical(net, l, weeks=(0,)):
    """Typical instances of link l, repeated over the given weeks."""
    tz, evs = net.links[l]
    out = []
    for q, w in ref_typical_events(evs):
        dep, ride = evs[q][0], evs[q][1]
        out += [(k * WEEK + w * DAY + dep, k * WEEK + w * DAY + dep + ride + tz,
                 int(net.ev0[l]) + q) for k in weeks]
    return sorted(out)


def ref_opportunities(lists, trip, windows, horizon=None):
    """Opportunities as sorted (first departure, last arrival, transfer
    minutes); with a horizon only first departures in [0, horizon) are
    returned (every itinerary can still beat them). Also the number of
    feasible itineraries."""
    found = []

    def walk(chain, gaps):
        level = len(chain)
        if level == len(lists):
            found.append((tuple(chain), tuple(gaps)))
            return
        for q, (dep, _, row) in enumerate(lists[level]):
            if level == 0:
                walk([q], [])
                continue
            _, arr, prev = lists[level - 1][chain[-1]]
            lo, move, hi = windows[level - 1]
            if lo + move <= dep - arr <= hi and trip[prev] != trip[row]:
                walk(chain + [q], gaps + [dep - arr])

    walk([], [])
    # one itinerary per (first departure, last arrival): the latest instance,
    # from the last link backwards
    points = {}
    for chain, gaps in found:
        key = (lists[0][chain[0]][0], lists[-1][chain[-1]][1])
        if key not in points or chain[::-1] > points[key][0]:
            points[key] = (chain[::-1], gaps)
    front = [k for k in points
             if (horizon is None or 0 <= k[0] < horizon)
             and not any(o != k and o[0] >= k[0] and o[1] <= k[1] for o in points)]
    return sorted((k[0], k[1], points[k][1]) for k in front), len(found)


def ref_stats(opps):
    """(n_days, freq, [ft_mean, ft_min per transfer ...], elapsed)."""
    if not opps:
        return None
    per_day = Counter(first // DAY for first, _, _ in opps)
    out = [len(per_day), statistics.median(per_day.values())]
    for t in range(len(opps[0][2])):
        out += [statistics.fmean(o[2][t] for o in opps), min(o[2][t] for o in opps)]
    return out + [statistics.fmean(last - first for first, last, _ in opps)]


def check_stats(got, opps):
    want = ref_stats(opps)
    if want is None:
        assert got[0] == 0 and got[1] == 0.0 and all(math.isnan(x) for x in got[2:])
    else:
        assert got[0] == want[0]
        assert list(got[1:]) == pytest.approx(want[1:], rel=1e-12)


def kernel_opps(first, last, ft):
    return [(int(f), int(a), tuple(int(x) for x in gaps))
            for f, a, gaps in zip(first, last, ft.T)]


def random_net(rng, n_links):
    """Links around successive hub times, with ties, shared trip keys, runs
    past midnight, time-zone shifts and irregular calendars."""
    span = int(rng.integers(8, 22))
    start = int(rng.integers(0, D.N_DAYS - span))
    links, windows = [], []
    base = int(rng.integers(0, DAY))
    for _ in range(n_links):
        tz = int(rng.choice([-60, 0, 0, 60]))
        ride0 = int(rng.choice([25, 60, 140, 300]))
        evs = []
        for _ in range(int(rng.integers(1, 7))):
            kind = int(rng.integers(0, 7))
            if kind < 2:
                days = list(range(start, start + span))
            elif kind < 4:
                days = [d for d in range(start, start + span) if rng.random() < 0.6]
            elif kind < 6:
                keep = rng.choice(7, size=int(rng.integers(1, 6)), replace=False)
                days = [d for d in range(start, start + span) if d % 7 in keep]
            else:
                days = []
            days = days or [start + int(rng.integers(0, span))]
            evs.append(((base + 5 * int(rng.integers(-8, 9))) % DAY,
                        ride0 + int(rng.choice([0, 0, 3, 10])),
                        int(rng.integers(0, 6)), days))
        links.append((tz, evs))
        lo, hi = WINDOWS[int(rng.integers(0, len(WINDOWS)))]
        windows.append((lo, float(rng.choice([0, 0, 3.5, 10, 25])), hi))
        base += ride0 + tz + lo + int(rng.integers(0, hi - lo + 1))
    return Net(*links), windows[:-1]


# --- Instances and typical days ----------------------------------------------
def test_instances_match_reference():
    rng = np.random.default_rng(11)
    for _ in range(20):
        net, _ = random_net(rng, 3)
        for kind, inst in (("dated", net.inst), ("typical", net.typ)):
            for l in range(3):
                want = ref_dated(net, l) if kind == "dated" else ref_typical(net, l)
                s, e = inst.ptr[l], inst.ptr[l + 1]
                got = list(zip(inst.dep[s:e].tolist(), inst.arr[s:e].tolist(),
                               inst.ev[s:e].tolist()))
                assert got == want
                rides = [r + net.links[l][0] for _, r, _, _ in net.links[l][1]]
                assert (inst.dmin[l], inst.dmax[l]) == (min(rides), max(rides))
        # one link on its own: same instances, rows counted from its first event
        s, e = net.ev0[1], net.ev1[1]
        dep, arr, ev = T.link_instances(net.dep[s:e], net.ride[s:e],
                                        net.masks[s:e], int(net.tz[1]))
        view = T.link_view(net.inst, 1)
        assert (dep == view.dep).all() and (arr == view.arr).all()
        assert (ev + s == view.ev).all()


def test_instances_layout_and_memory():
    net = Net((60, [(hm(23, 30), 45, 1, [0, 5]), (hm(6), 50, 2, [5])]))
    dep, arr, ev = net.inst.dep, net.inst.arr, net.inst.ev
    assert dep.tolist() == [hm(23, 30), 5 * DAY + hm(6), 5 * DAY + hm(23, 30)]
    assert arr.tolist() == [DAY + 75, 5 * DAY + hm(7, 50), 6 * DAY + 75]
    assert ev.tolist() == [0, 1, 0]
    assert dep.dtype == arr.dtype == ev.dtype == np.int32
    assert T.instance_bytes(3, 1) == sum(x.nbytes for x in net.inst)
    assert T.instance_bytes(48_000_000, 70_000) < 0.6 * 2 ** 30
    with pytest.raises(ValueError):
        T.build_instances([1440], [10], net.masks[:1], [0], [1], [0])


def test_typical_days_share():
    mondays = [MONDAY + 7 * k for k in range(6)]
    tuesdays = [d + 1 for d in mondays]
    net = Net((0, [(hm(8), 60, 1, mondays + tuesdays[:2]),     # 6 of 6 Mon, 2 of 6 Tue
                   (hm(9), 60, 2, mondays[:3] + tuesdays[1:]),  # 3 of 6 Mon, 5 of 6 Tue
                   (hm(10), 60, 3, mondays[:2])]))              # 2 of 6 Mon
    assert SHARE == 0.5
    assert T.typical_flags(net.masks, net.ev0, net.ev1).tolist() == [0b01, 0b11, 0]
    assert net.typ.dep.tolist() == [hm(8), hm(9), DAY + hm(9)]
    assert net.typ.arr.tolist() == [hm(9), hm(10), DAY + hm(10)]
    assert net.typ.ev.tolist() == [0, 1, 1]


def season_days(s):
    return [d for d in ALL_DAYS if SEASON[d] == s]


def test_typical_week_follows_one_season():
    """One flight a day whose clock time depends on the weekday and changes
    every season, as air link 69184 of the 13 Nov network: over the whole
    timeline no version runs on half of a weekday's dates, within its season
    each does. Every season matches the link's week of 7; the longest wins."""
    seasons = range(max(SEASON) + 1)
    evs = [(hm(6) + 25 * s + 10 * w, 85 + 5 * (s % 2), 100 + 7 * s + w,
            [d for d in season_days(s) if WD[d] == w]) for s in seasons for w in range(7)]
    net = Net((0, evs), (0, [(hm(12), 120, 1, ALL_DAYS)]))
    longest = max(seasons, key=lambda s: (len(season_days(s)), s))
    s, e = net.typ.ptr[0], net.typ.ptr[1]
    assert net.typ.dep[s:e].tolist() == [w * DAY + hm(6) + 25 * longest + 10 * w
                                         for w in range(7)]
    assert list(zip(net.typ.dep[s:e].tolist(), net.typ.arr[s:e].tolist(),
                    net.typ.ev[s:e].tolist())) == ref_typical(net, 0)
    assert net.pair(5, 360)[:2] == (D.N_DAYS, 1)
    assert net.pair(5, 360, basis="T")[:2] == (7, 1)


def test_typical_season_closest_to_own_week():
    # 08:00 daily and a 12:00 extra in one winter: that winter's week (14) is
    # further from the link's week (7) than the others, and of those the
    # longest season wins
    net = Net((0, [(hm(8), 60, 1, ALL_DAYS), (hm(12), 60, 2, season_days(2))]))
    assert T.typical_flags(net.masks, net.ev0, net.ev1).tolist() == [0x7F, 0]
    # two seasons of equal length and equal fit: the later one
    assert len(season_days(3)) == len(season_days(5))
    net = Net((0, [(hm(8), 60, 1, season_days(3)), (hm(9), 60, 2, season_days(5))]))
    assert T.typical_flags(net.masks, net.ev0, net.ev1).tolist() == [0, 0x7F]
    # a weekly flight: the link's week is 1, not 7 times the median per date
    sundays = [d for d in ALL_DAYS if WD[d] == 6]
    net = Net((0, [(hm(8), 60, 1, sundays), (hm(10), 60, 2, sundays[:20])]))
    assert T.typical_flags(net.masks, net.ev0, net.ev1).tolist() == [1 << 6, 0]


def test_typical_matches_reference_across_seasons():
    """Versions that start and stop at random dates around season changes."""
    rng = np.random.default_rng(17)
    starts = [d for d in range(1, D.N_DAYS) if SEASON[d] != SEASON[d - 1]]
    links = 0
    for _ in range(40):
        b = starts[int(rng.integers(len(starts)))]
        first = max(0, b - int(rng.integers(10, 300)))
        last = min(D.N_DAYS, b + int(rng.integers(10, 300)))
        evs = []
        for q in range(int(rng.integers(1, 9))):
            a = int(rng.integers(first, last))
            z = int(rng.integers(a + 1, last + 1))
            keep = set(rng.choice(7, size=int(rng.integers(1, 8)), replace=False).tolist())
            days = [d for d in range(a, z) if WD[d] in keep and rng.random() < 0.9] or [a]
            evs.append((hm(7) + 5 * int(rng.integers(0, 6)), 60, q, days))
        net = Net((0, evs), (60, evs[::-1]))
        for l in range(2):
            s, e = net.typ.ptr[l], net.typ.ptr[l + 1]
            got = list(zip(net.typ.dep[s:e].tolist(), net.typ.arr[s:e].tolist(),
                           net.typ.ev[s:e].tolist()))
            assert got == ref_typical(net, l)
            links += e > s
    assert links >= 60


# --- Windows -----------------------------------------------------------------
@pytest.mark.parametrize("lo,hi,gaps", [(5, 60, (4, 5, 60, 61)),
                                        (45, 180, (44, 45, 180, 181))])
def test_window_edges(lo, hi, gaps):
    for basis in "DT":
        for gap, feasible in zip(gaps, (False, True, True, False)):
            net = daily((0, [(hm(8), 120, 1)]), (0, [(hm(10) + gap, 30, 2)]))
            n_days, freq, ft_mean, ft_min, elapsed = net.pair(lo, hi, basis=basis)
            if feasible:
                assert n_days == (D.N_DAYS if basis == "D" else 7) and freq == 1
                assert ft_mean == ft_min == gap and elapsed == 150 + gap
            else:
                assert n_days == 0 and freq == 0 and math.isnan(ft_mean)


def test_move_shifts_lower_bound():
    def feasible(gap, move):
        net = daily((0, [(hm(8), 120, 1)]), (0, [(hm(10) + gap, 30, 2)]))
        days = (net.pair(5, 60, move)[0], net.pair(5, 60, move, "T")[0])
        assert days in ((0, 0), (D.N_DAYS, 7))
        return days[0] > 0

    assert not feasible(14, 10) and feasible(15, 10)       # 5 + 10
    assert not feasible(12, 7.5) and feasible(13, 7.5)     # 5 + 7.5
    assert feasible(60, 55) and not feasible(61, 55)       # the upper bound stays
    assert not feasible(60, 56)                            # no room left


def test_same_trip_is_not_a_transfer():
    through = daily((0, [(hm(8), 120, 7)]), (0, [(hm(10, 10), 30, 7)]))
    assert through.pair(5, 60)[0] == 0 and through.pair(5, 60, basis="T")[0] == 0
    # the next train is a transfer, and the through train does not hide it
    both = daily((0, [(hm(8), 120, 7)]),
                 (0, [(hm(10, 10), 30, 7), (hm(10, 20), 30, 8)]))
    for basis in "DT":
        assert both.pair(5, 60, basis=basis)[1:4] == (1, 20, 20)


# --- Clocks ------------------------------------------------------------------
def test_midnight_rollover():
    days = range(100, 110)
    # arrival 23:50, onward 00:10 the next date
    net = Net((0, [(hm(23), 50, 1, days)]), (0, [(hm(0, 10), 30, 2, days)]))
    assert net.pair(5, 60) == (9, 1, 20, 20, 100)        # the last evening has no onward run
    assert net.pair(5, 60, basis="T") == (7, 1, 20, 20, 100)
    # the first run itself passes midnight: 22:00 + 3 h, onward 01:15
    net = Net((0, [(hm(22), 180, 1, days)]), (0, [(hm(1, 15), 30, 2, days)]))
    assert net.pair(5, 60) == (9, 1, 15, 15, 225)
    assert net.pair(5, 60, basis="T") == (7, 1, 15, 15, 225)


def test_sunday_to_monday():
    mondays = [MONDAY + 7 * k for k in range(1, 9)]
    sundays = [d - 1 for d in mondays]
    net = Net((0, [(hm(23), 50, 1, sundays)]), (0, [(hm(0, 10), 30, 2, mondays)]))
    assert net.pair(5, 60) == (8, 1, 20, 20, 100)
    assert net.pair(5, 60, basis="T") == (1, 1, 20, 20, 100)
    # no common date: the default basis is typical days
    assert chr(T.pair_stats(net.inst, net.typ, net.cal, net.trip, 0, 1, 0.0, 5, 60,
                            MIN_COMMON, False, net.work)[0]) == "T"
    # Monday evening does not reach Sunday night
    back = Net((0, [(hm(23), 50, 1, mondays)]), (0, [(hm(0, 10), 30, 2, sundays)]))
    assert back.pair(5, 60)[0] == 0 and back.pair(5, 60, basis="T")[0] == 0


def test_tz_shift():
    # +60: 08:00 + 60 min arrives at 10:00 on the clock of the alight hub
    east = daily((60, [(hm(8), 60, 1)]), (0, [(hm(9, 5), 30, 2), (hm(10, 5), 30, 3)]))
    # -60: the same run arrives at 08:00
    west = daily((-60, [(hm(8), 60, 1)]), (0, [(hm(8, 5), 30, 2), (hm(7, 55), 30, 3)]))
    for basis in "DT":
        assert east.pair(5, 60, basis=basis)[1:] == (1, 5, 5, hm(2, 35))
        assert west.pair(5, 60, basis=basis)[1:] == (1, 5, 5, 35)
    # -60 across midnight: Monday 00:10 + 30 min arrives Sunday 23:40
    mondays = [MONDAY + 7 * k for k in range(1, 9)]
    sundays = [d - 1 for d in mondays]
    net = Net((-60, [(hm(0, 10), 30, 1, mondays)]), (0, [(hm(23, 50), 20, 2, sundays)]))
    assert net.pair(5, 60) == (8, 1, 10, 10, 0)
    assert net.pair(5, 60, basis="T") == (1, 1, 10, 10, 0)


# --- Opportunities -----------------------------------------------------------
def test_feeder_bottleneck():
    """Four feeders into one daily trunk: one opportunity a day, and its
    transfer time is the last feeder's."""
    feeders = [(hm(8) + 15 * k, 60, k) for k in range(4)]       # arrive 9:00 .. 9:45
    net = daily((0, feeders), (0, [(hm(10), 120, 9)]))
    for basis in "DT":
        n_days, freq, ft_mean, ft_min, elapsed = net.pair(5, 60, basis=basis)
        assert freq == 1 and ft_mean == ft_min == 15 and elapsed == 195
    first, last, ft = T.opportunities2(T.link_view(net.inst, 0),
                                       T.link_view(net.inst, 1), net.trip, 5, 60)
    assert len(first) == D.N_DAYS and (first % DAY == hm(8, 45)).all()
    # one feeder and four onward runs: only the first to arrive counts
    fan = daily((0, [(hm(8), 60, 9)]), (0, [(hm(9, 10) + 10 * k, 60, k) for k in range(4)]))
    assert fan.pair(5, 60)[1:] == (1, 10, 10, 130)


def test_missed_first_connection():
    """Three links. The 08:00 run connects to the 09:10, which arrives 70 min
    before the only third run and misses it; the 09:00 run makes it through
    the 10:10."""
    net = daily((0, [(hm(8), 60, 1), (hm(9), 60, 2)]),
                (0, [(hm(9, 10), 50, 3), (hm(10, 10), 50, 4)]),
                (0, [(hm(11, 10), 40, 5)]))
    assert net.pair(5, 60)[1] == 2                 # two opportunities on the first pair
    for basis in "DT":
        n_days, freq, m1, low1, m2, low2, elapsed = net.triple((5, 0, 60), (5, 0, 60), basis)
        assert n_days == (D.N_DAYS if basis == "D" else 7)
        assert (freq, m1, low1, m2, low2, elapsed) == (1, 10, 10, 10, 10, 170)
    first = T.opportunities3(*(T.link_view(net.inst, l) for l in range(3)),
                             net.trip, 5, 60, 5, 60)[0]
    assert (first % DAY == hm(9)).all()
    # the other way round: the later first run misses, the earlier one makes it
    net = daily((0, [(hm(8), 60, 1), (hm(8, 30), 60, 2)]),
                (0, [(hm(9, 20), 60, 3)]),
                (0, [(hm(10, 30), 40, 5)]))
    for basis in "DT":
        assert net.triple((5, 0, 60), (5, 0, 60), basis)[1:] == (1, 20, 20, 10, 10, 190)


def test_identical_itineraries_count_once():
    # two feeders and two onward runs with the same clock times
    net = daily((0, [(hm(8), 60, 1), (hm(8), 60, 2)]),
                (0, [(hm(9, 20), 30, 3), (hm(9, 20), 30, 4)]))
    assert net.pair(5, 60)[1:] == (1, 20, 20, 110)
    assert net.pair(5, 60, basis="T")[1:] == (1, 20, 20, 110)


# --- Basis -------------------------------------------------------------------
def test_dated_and_typical_agree_on_regular_timetable():
    def same(x, y, rel=0.0):
        return list(x) == pytest.approx(list(y), rel=rel, abs=0.0, nan_ok=True)

    rng = np.random.default_rng(5)
    chains = 0
    for case in range(20):
        # daytime timetables first, then hub times anywhere on the clock
        hub = hm(9) if case < 10 else int(rng.integers(0, DAY))
        evs = [[((hub + 150 * l + int(rng.integers(0, 120))) % DAY,
                 60 + int(rng.integers(0, 8)), 10 * l + q)
                for q in range(int(rng.integers(1, 6)))] for l in range(3)]
        net = daily(*[(0, e) for e in evs])
        w = (5, 0.0, 120)
        dated, typical = net.pair(5, 120), net.pair(5, 120, basis="T")
        dated3, typical3 = net.triple(w, w), net.triple(w, w, "T")
        chains += dated3[0] > 0
        if case < 10:
            # nothing passes midnight: the two bases are identical
            assert same(dated[1:], typical[1:]) and same(dated3[1:], typical3[1:])
            assert (dated[0], typical[0]) in ((D.N_DAYS, 7), (0, 0))
            assert (dated3[0], typical3[0]) in ((D.N_DAYS, 7), (0, 0))
        else:
            # the last day of the timeline may lose its overnight connection
            assert dated[1] == typical[1] and dated3[1] == typical3[1]
            assert same(dated[2:], typical[2:], 5e-3)
            assert same(dated3[2:], typical3[2:], 5e-3)
    assert chains >= 15


def test_basis_follows_common_dates():
    def net(common):
        return Net((0, [(hm(8), 120, 1, range(200, 240))]),
                   (0, [(hm(10, 10), 30, 2, range(240 - common, 300))]),
                   (0, [(hm(10, 50), 30, 3, range(240 - common, 300))]))

    for common, basis in ((MIN_COMMON - 1, "T"), (MIN_COMMON, "D")):
        n = net(common)
        out = T.transfer_table(n.inst, n.typ, n.cal, n.trip, [0, 0], [1, 1],
                               [0.0, 0.0], [5, 5], [60, 60], typical=[False, True])
        assert T.basis_labels(out["basis"]).tolist() == [basis, "T"]
        assert out["n_days"].tolist() == [common if basis == "D" else 7, 7]
        triple = T.triple_stats(n.inst, n.typ, n.cal, n.trip, 0, 1, 2, 0.0, 5, 60,
                                0.0, 5, 60, MIN_COMMON, False, n.work)
        assert chr(triple[0]) == basis


def test_transfer_table_equals_single_pairs(monkeypatch):
    monkeypatch.setattr(T, "CHUNK", 7)               # several calls of the kernel
    rng = np.random.default_rng(3)
    net, _ = random_net(rng, 6)
    n = 40
    a, b = rng.integers(0, 6, n), rng.integers(0, 6, n)
    move = rng.choice([0, 3.5, 10], n).astype(np.float32)
    lo, hi = np.array([WINDOWS[i] for i in rng.integers(0, len(WINDOWS), n)]).T
    forced = rng.random(n) < 0.3
    out = T.transfer_table(net.inst, net.typ, net.cal, net.trip, a, b, move, lo, hi,
                           typical=forced)
    assert [out[c].dtype for c in ("basis", "n_days", "freq", "ft_mean", "ft_min")] == [
        np.uint8, np.int32, np.float32, np.float32, np.float32]
    for p in range(n):
        want = T.pair_stats(net.inst, net.typ, net.cal, net.trip, int(a[p]), int(b[p]),
                            float(move[p]), int(lo[p]), int(hi[p]), MIN_COMMON,
                            bool(forced[p]), net.work)
        got = [out[c][p] for c in ("basis", "n_days", "freq", "ft_mean", "ft_min")]
        assert got == pytest.approx(list(want[:5]), rel=1e-6, nan_ok=True)


def test_transfer_table_rejects_bad_input():
    net = daily((0, [(hm(8), 60, 1)]), (0, [(hm(9, 10), 60, 2)]))
    good = dict(typ=net.typ, cal=net.cal, trip=net.trip, link_in=[0], link_out=[1],
                move=[0.0], lo=[5], hi=[60], typical=[False])

    def call(**bad):
        kw = {**good, **bad}
        return T.transfer_table(net.inst, kw.pop("typ"), kw.pop("cal"), kw.pop("trip"), **kw)

    assert call()["n_days"].tolist() == [D.N_DAYS]
    one_link = T.build_typical(net.dep, net.ride, net.masks, net.ev0[:1], net.ev1[:1],
                               net.tz[:1])
    for bad in ({"link_out": [2]}, {"link_in": [-1]}, {"move": [0.0, 0.0]},
                {"hi": [60, 60]}, {"typical": [True, False]}, {"lo": [61]},
                {"cal": net.cal[:1]}, {"trip": net.trip[:1]}, {"typ": one_link}):
        with pytest.raises(ValueError):
            call(**bad)


@njit
def _search_like(inst, typ, cal, trip, chains, windows, min_common, work, out):
    """A compiled loop as the search will write it: triple_stats per chain."""
    for q in range(chains.shape[0]):
        basis, n_days, freq, m1, low1, m2, low2, elapsed = T.triple_stats(
            inst, typ, cal, trip, chains[q, 0], chains[q, 1], chains[q, 2],
            windows[q, 1], int(windows[q, 0]), int(windows[q, 2]),
            windows[q, 4], int(windows[q, 3]), int(windows[q, 5]),
            min_common, q % 2 == 1, work)
        out[q, 0] = basis
        out[q, 1] = n_days
        out[q, 2:] = np.array([freq, m1, low1, m2, low2, elapsed])


def test_triple_stats_inside_compiled_loop():
    rng = np.random.default_rng(13)
    net, (w1, w2) = random_net(rng, 3)
    chains = np.array([[0, 1, 2]] * 6, dtype=np.int64)
    windows = np.array([[*w1, *w2]] * 6, dtype=np.float64)
    out = np.empty((6, 8))
    _search_like(net.inst, net.typ, net.cal, net.trip, chains, windows,
                 MIN_COMMON, net.work, out)
    for q in range(6):
        want = T.triple_stats(net.inst, net.typ, net.cal, net.trip, 0, 1, 2,
                              w1[1], w1[0], w1[2], w2[1], w2[0], w2[2], MIN_COMMON,
                              q % 2 == 1, net.work)
        assert list(out[q]) == pytest.approx(list(want), abs=0, nan_ok=True)


def test_workspace_size_does_not_change_results():
    rng = np.random.default_rng(8)
    tiny = np.empty((T.ROWS, 1), dtype=np.int32)
    for _ in range(10):
        net, (w1, w2) = random_net(rng, 3)
        for forced in (False, True):
            args = (net.inst, net.typ, net.cal, net.trip, 0, 1, 2, w1[1], w1[0], w1[2],
                    w2[1], w2[0], w2[2], 0, forced)
            assert list(T.triple_stats(*args, tiny)) == pytest.approx(
                list(T.triple_stats(*args, net.work)), abs=0, nan_ok=True)
            args = (net.inst, net.typ, net.cal, net.trip, 0, 1, w1[1], w1[0], w1[2], 0, forced)
            assert list(T.pair_stats(*args, tiny)) == pytest.approx(
                list(T.pair_stats(*args, net.work)), abs=0, nan_ok=True)
    assert net.work.shape == (T.ROWS, T.workspace_size(net.inst, net.typ))


# --- Random cases against the reference --------------------------------------
REF_WEEKS = range(-1, 4)      # more than any chain of the random cases can span


@pytest.mark.parametrize("n_links,seed", [(2, 2026), (3, 2027)])
def test_kernels_equal_reference(n_links, seed):
    rng = np.random.default_rng(seed)
    with_opps = filtered = trips_cut = 0
    for _ in range(200):
        net, windows = random_net(rng, n_links)
        gmin_hi = [x for lo, move, hi in windows for x in (math.ceil(lo + move), hi)]
        views = [T.link_view(net.inst, l) for l in range(n_links)]
        kernel = T.opportunities2 if n_links == 2 else T.opportunities3

        def stats(basis):
            if n_links == 3:
                return net.triple(*windows, basis)
            lo, move, hi = windows[0]
            return net.pair(lo, hi, move, basis)

        # date by date
        lists = [ref_dated(net, l) for l in range(n_links)]
        want, n_feasible = ref_opportunities(lists, net.trip, windows)
        assert kernel_opps(*kernel(*views, net.trip, *gmin_hi)) == want
        check_stats(stats("D"), want)
        with_opps += bool(want)
        filtered += n_feasible > len(want)
        trips_cut += n_feasible < ref_opportunities(
            lists, np.arange(len(net.trip)), windows)[1]
        # typical days
        lists = [ref_typical(net, l, REF_WEEKS) for l in range(n_links)]
        want, _ = ref_opportunities(lists, net.trip, windows, WEEK)
        check_stats(stats("T"), want)
    # the cases do exercise the rules
    assert with_opps >= 150 and filtered >= 120 and trips_cut >= 80
