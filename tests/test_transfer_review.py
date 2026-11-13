"""Reviewer's cases for pathset/transfer.py.

1. Random chains outside the range of test_transfer.random_net: riding times
   that differ by up to 200 min within a link, tz_shift of -120..+120 (so an
   arrival can fall before its departure and before minute 0 of the week),
   every window of WINDOWS with moves up to AIR_ROAD_MOVE_MAX_MIN, up to 11
   events per link. Kernels against the brute-force reference of
   test_transfer, date by date and on typical days.
2. A daily service whose departure minute is retimed between timetable periods
   keeps one typical departure a day. Before the fix (share test over the whole
   timeline, per exact (dep, ride) event) it had none.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_transfer as TT  # noqa: E402
from pathset import dates as D  # noqa: E402
from pathset import transfer as T  # noqa: E402
from pathset.io import P  # noqa: E402

DAY, WEEK = 1440, 10080
WINDOWS = [tuple(w) for w in P["WINDOWS"].values()]
MOVES = [0.0, 0.0, 0.5, 7.25, float(P["MOVE_MAX_MIN"]), float(P["AIR_ROAD_MOVE_MAX_MIN"])]


def wide_net(rng, n_links):
    """Links with wide riding-time spreads, large time-zone shifts and moves."""
    span = int(rng.integers(8, 40))
    start = int(rng.integers(0, D.N_DAYS - span))
    links, windows = [], []
    base = int(rng.integers(0, DAY))
    for _ in range(n_links):
        tz = int(rng.choice([-120, -60, 0, 60, 120]))
        ride0 = int(rng.choice([10, 25, 60, 140, 300, 700]))
        evs = []
        for _ in range(int(rng.integers(1, 12))):
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
            evs.append(((base + int(rng.integers(-120, 121))) % DAY,
                        max(1, ride0 + int(rng.choice([0, 0, 2, 30, 200, -5]))),
                        int(rng.integers(0, 4)), days))
        links.append((tz, evs))
        lo, hi = WINDOWS[int(rng.integers(0, len(WINDOWS)))]
        windows.append((lo, MOVES[int(rng.integers(0, len(MOVES)))], hi))
        base += ride0 + tz + lo + int(rng.integers(0, hi - lo + 1))
    return TT.Net(*links), windows[:-1]


@pytest.mark.parametrize("n_links,seed", [(2, 501), (3, 502)])
def test_wide_spreads_equal_reference(n_links, seed):
    rng = np.random.default_rng(seed)
    with_opps = 0
    for _ in range(80):
        net, windows = wide_net(rng, n_links)
        gmin_hi = [x for lo, move, hi in windows for x in (math.ceil(lo + move), hi)]
        views = [T.link_view(net.inst, l) for l in range(n_links)]
        kernel = T.opportunities2 if n_links == 2 else T.opportunities3

        def stats(basis):
            if n_links == 3:
                return net.triple(*windows, basis)
            lo, move, hi = windows[0]
            return net.pair(lo, hi, move, basis)

        lists = [TT.ref_dated(net, l) for l in range(n_links)]
        want, _ = TT.ref_opportunities(lists, net.trip, windows)
        assert TT.kernel_opps(*kernel(*views, net.trip, *gmin_hi)) == want
        TT.check_stats(stats("D"), want)
        with_opps += bool(want)
        lists = [TT.ref_typical(net, l, range(-3, 7)) for l in range(n_links)]
        want, _ = TT.ref_opportunities(lists, net.trip, windows, WEEK)
        TT.check_stats(stats("T"), want)
    assert with_opps >= 30


def test_retimed_daily_service_survives_on_typical_days():
    thirds = np.array_split(np.arange(D.N_DAYS), 3)
    feeder = (0, [(TT.hm(8) + 5 * k, 60, 1, part.tolist())
                  for k, part in enumerate(thirds)])
    trunk = (0, [(TT.hm(9, 40), 120, 2, range(D.N_DAYS))])
    net = TT.Net(feeder, trunk)
    dated = net.pair(5, 60)
    assert dated[0] == D.N_DAYS and dated[1] == 1
    typical = net.pair(5, 60, basis="T")
    assert typical[0] == 7 and typical[1] == 1
