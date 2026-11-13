"""Independent review of pathset/network.py: brute-force comparisons of the
kernels and adversarial cases the module's own tests do not cover.

test_one_vehicle_keeps_one_trip_key_when_a_leg_is_shared reproduces a defect
found in review (trip keys of one vehicle differed once one of its legs was
merged into a run of another trip); ``vehicle_keys`` fixes it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import dates as D  # noqa: E402
from pathset import network as N  # noqa: E402

R, B, A = N.MODE_R, N.MODE_B, N.MODE_A
HUBS = pd.DataFrame({
    "hub": [0, 1, 2, 3, 4], "fua": [10, 10, 11, 12, 13],
    "tz_std": np.array([1, 1, 0, 2, 1], dtype=np.int8),
    "lon": [4.0, 4.02, 0.0, 25.0, 9.0], "lat": [52.0, 52.0, 51.5, 60.0, 48.0]})
WEEKDAY = D.weekdays()
WINTER = np.arange(8, 50)
MONDAYS = WINTER[WEEKDAY[WINTER] == 0]
RNG = np.random.default_rng(20261113)


def cal(*day_lists) -> np.ndarray:
    mat = np.zeros((len(day_lists), D.N_DAYS), dtype=bool)
    for i, days in enumerate(day_lists):
        mat[i, np.asarray(days, dtype=int)] = True
    return D.pack_bool(mat)


def days(mask) -> np.ndarray:
    return np.flatnonzero(D.unpack(np.atleast_2d(mask))[0])


def calls(*journeys) -> pd.DataFrame:
    return pd.DataFrame(
        [(j, h, a, d) for j, stops in journeys for h, a, d in stops],
        columns=["jrn", "hub", "arr", "dep"])


def journeys(n: int) -> pd.DataFrame:
    return pd.DataFrame({"jrn": np.arange(1, n + 1), "rail": True,
                         "route_type": 2, "cal": np.arange(n)})


def build(calls_df, journeys_df, masks):
    """Steps b to h of pathset-network.py on small tables."""
    trips = N.build_trips(calls_df, journeys_df, HUBS["tz_std"].to_numpy(), masks)
    legs, _ = N.build_legs(trips.calls, trips.trips["mode"].to_numpy(),
                           HUBS["fua"].to_numpy(), HUBS["tz_std"].to_numpy())
    runs, run_of_leg = N.merge_equal(legs)
    run_masks = N.or_calendars(run_of_leg, len(runs), legs["trip"],
                               legs["shift"], trips.masks)
    runs, run_masks, kept_of = N.merge_near(runs, run_masks, 2)
    cls = N.ride_classes(runs, 5, 0.05, True)
    net = N.build_links(runs, run_masks, cls, HUBS, 1, (0, D.N_DAYS - 1))
    event_of_leg = net.event_of_run[kept_of[run_of_leg]]
    net.events["trip"] = N.vehicle_keys(legs["trip"], event_of_leg,
                                        len(net.events))
    return net


# --- adversarial cases ---------------------------------------------------------
def test_leg_from_a_western_hub_departs_on_the_previous_local_date():
    # leaves hub 0 (UTC+1) at 00:20 CET on Mondays = 23:20 UTC on Sunday;
    # calls at hub 2 (UTC+0) 23:50-23:55 Sunday; reaches hub 3 (UTC+2) at
    # 03:40 EET on Monday
    c = calls((1, [(0, -40, -40), (2, -10, -5), (3, 100, 100)]))
    net = build(c, journeys(1), cal(MONDAYS))
    ev = net.events.merge(net.links[["link", "src", "trg"]], on="link")
    ev["weekdays"] = [set(WEEKDAY[days(m)].tolist()) for m in net.masks]
    by = ev.set_index(["src", "trg"])
    assert by.loc[(0, 2), "dep"] == 20 and by.loc[(0, 2), "weekdays"] == {0}
    assert by.loc[(2, 3), "dep"] == 1435 and by.loc[(2, 3), "weekdays"] == {6}
    assert by.loc[(2, 3), "ride"] == 105 and by.loc[(0, 3), "ride"] == 140


def test_one_vehicle_keeps_one_trip_key_when_a_leg_is_shared():
    # Trip T1: hub 0 -> hub 2 -> hub 3 on Mondays. Trip T2: the same times
    # from hub 0 to hub 2 (shared run), then on to hub 4 on Tuesdays,
    # dwelling 10 min at hub 2.
    c = calls((1, [(0, 500, 500), (2, 600, 610), (3, 700, 700)]),
              (2, [(0, 500, 500), (2, 600, 610), (4, 720, 720)]))
    net = build(c, journeys(2), cal(MONDAYS, MONDAYS + 1))
    ev = net.events.merge(net.links[["link", "src", "trg"]], on="link")
    key = ev.set_index(["src", "trg"])["trip"]
    # T2 arrives at hub 2 on run (0, 2) and continues on run (2, 4): the same
    # vehicle, so the transfer rule "trip keys differ" must not admit it
    assert key.loc[(0, 2)] == key.loc[(2, 4)]


# --- brute-force comparisons of the kernels ----------------------------------
def test_link_stats_against_a_dense_recount():
    n = 60
    runs = pd.DataFrame({
        "src": RNG.integers(0, 2, n) * 2, "trg": 3, "mode": R,
        "dep": RNG.integers(0, 1440, n), "ride": RNG.integers(50, 80, n)})
    runs = runs.drop_duplicates(["src", "trg", "mode", "dep", "ride"])
    runs = runs.reset_index(drop=True)
    runs["trip"] = np.arange(len(runs))
    mat = RNG.random((len(runs), D.N_DAYS)) < 0.02
    mat[:, 0] = True
    masks = D.pack_bool(mat)
    runs["n_dates"] = mat.sum(1)
    cls = N.ride_classes(runs, 5, 0.05, True)
    fixed = (100, 499)
    net = N.build_links(runs, masks, cls, HUBS, 1, fixed)
    runs["cls"] = cls
    for l in net.links.itertuples():
        sel = ((runs["src"] == l.src) & (runs["cls"] == l.cls)).to_numpy()
        count = mat[sel].sum(0)
        act = np.flatnonzero(count)
        assert l.freq == pytest.approx(np.median(count[act]))
        assert l.freq_max == count.max() and l.active_dates == len(act)
        assert (l.first_day, l.last_day) == (act[0], act[-1])
        w = runs.loc[sel, "n_dates"].to_numpy()
        assert l.ride == pytest.approx(
            (runs.loc[sel, "ride"].to_numpy() * w).sum() / w.sum(), rel=1e-6)
        assert l.freq_fixed == pytest.approx(count[100:500].sum() / 400, rel=1e-6)
        span = act[-1] - act[0] + 1
        assert l.freq_span == pytest.approx(count.sum() / span, rel=1e-6)
        assert l.active_share == pytest.approx(len(act) / span, rel=1e-6)
        rows = np.arange(l.ev0, l.ev1)
        assert (net.events["link"].to_numpy()[rows] == l.link).all()
        assert sorted(D.popcount_rows(net.masks[rows]).tolist()) == sorted(
            mat[sel].sum(1).tolist())


def brute_classes(rides, abs_min, rel):
    out, c, anchor = {}, -1, None
    for r in sorted(set(rides)):
        if anchor is None or r > anchor + max(abs_min, rel * anchor):
            c, anchor = c + 1, r
        out[r] = c
    return [out[r] for r in rides]


def test_ride_classes_against_brute_force():
    for _ in range(20):
        rides = RNG.integers(20, 400, 30).tolist()
        runs = pd.DataFrame({"src": 0, "trg": 2, "mode": R,
                             "dep": np.arange(30), "ride": rides})
        assert N.ride_classes(runs, 5, 0.05, True).tolist() == brute_classes(
            rides, 5, 0.05)


def test_weekday_calendars_against_pandas():
    dates = pd.date_range(D.DAY0, D.DAY1)
    for _ in range(30):
        d0, d1 = sorted(RNG.integers(-5, D.N_DAYS + 5, 2))
        bits = int(RNG.integers(0, 128))
        got = N.weekday_calendars([0], 1, [max(d0, 0)], [min(d1, D.N_DAYS - 1)],
                                  np.array([bits], dtype=np.uint8))
        want = [i for i, d in enumerate(dates)
                if max(d0, 0) <= i <= min(d1, D.N_DAYS - 1)
                and bits >> d.dayofweek & 1]
        assert days(got[0]).tolist() == want


def test_signatures_against_a_dictionary():
    seqs = [tuple(map(tuple, RNG.integers(0, 3, (RNG.integers(2, 4), 3))))
            for _ in range(200)]
    start = np.r_[0, np.cumsum([len(s) for s in seqs])]
    flat = np.array([c for s in seqs for c in s])
    sig = N.signatures(start, flat[:, 0], flat[:, 1], flat[:, 2])
    first = {}
    want = [first.setdefault(s, len(first)) for s in seqs]
    assert sig.tolist() == want


def test_near_merge_keeps_every_date_and_stays_within_tolerance():
    for _ in range(10):
        n = 80
        runs = pd.DataFrame({"src": 0, "trg": 2, "mode": R,
                             "dep": RNG.integers(600, 620, n),
                             "ride": RNG.integers(60, 66, n)})
        runs = (runs.drop_duplicates().sort_values(["dep", "ride"])
                .reset_index(drop=True))
        runs["trip"] = np.arange(len(runs))
        mat = RNG.random((len(runs), 400)) < 0.03
        full = np.zeros((len(runs), D.N_DAYS), dtype=bool)
        full[:, :400] = mat
        masks = D.pack_bool(full)
        out, out_masks, kept_of = N.merge_near(runs, masks, 2)
        # no date is lost or invented
        union_in = D.unpack(masks)
        union_out = D.unpack(out_masks)
        for k in range(len(out)):
            assert (union_out[k] == union_in[kept_of == k].any(0)).all()
        # every absorbed run lies within the tolerance of its kept run
        assert (np.abs(runs["dep"].to_numpy()
                       - out["dep"].to_numpy()[kept_of]) <= 2).all()
        assert (np.abs(runs["ride"].to_numpy()
                       - out["ride"].to_numpy()[kept_of]) <= 2).all()
