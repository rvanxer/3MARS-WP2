"""Static service network: hubs, trips, runs, classes, frequencies, car.

Every table is synthetic and built here. Hubs of the service tests:
hub 0 and 1 in FUA 10 (UTC+1), hub 2 in FUA 11 (UTC+0), hub 3 in FUA 12
(UTC+2), hub 4 in FUA 13 (UTC+1).
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
WINTER = np.arange(8, 50)                       # 9 January - 19 February 2023
MONDAYS = WINTER[WEEKDAY[WINTER] == 0]
JULY = np.arange(D.day_index(pd.Timestamp("2023-07-03").date()),
                 D.day_index(pd.Timestamp("2023-07-17").date()))
ALL_DAYS = np.arange(D.N_DAYS)


def cal(*day_lists) -> np.ndarray:
    """One calendar per list of day indices."""
    mat = np.zeros((len(day_lists), D.N_DAYS), dtype=bool)
    for i, days in enumerate(day_lists):
        mat[i, np.asarray(days, dtype=int)] = True
    return D.pack_bool(mat)


def days(mask) -> np.ndarray:
    return np.flatnonzero(D.unpack(np.atleast_2d(mask))[0])


def calls(*journeys) -> pd.DataFrame:
    """journeys: (jrn, [(hub, arr, dep), ...]) in exported minutes."""
    return pd.DataFrame(
        [(j, h, a, d) for j, stops in journeys for h, a, d in stops],
        columns=["jrn", "hub", "arr", "dep"])


def journeys(n: int, rail=True, route_type=2) -> pd.DataFrame:
    """Journeys 1..n; journey i uses calendar row i - 1."""
    return pd.DataFrame({"jrn": np.arange(1, n + 1), "rail": rail,
                         "route_type": route_type, "cal": np.arange(n)})


def runs_of(rows) -> pd.DataFrame:
    """Runs from (src, trg, mode, dep, ride) tuples; trip = row number."""
    df = pd.DataFrame(rows, columns=N.RUN_KEY)
    df["trip"] = np.arange(len(df))
    return df


def links_of(runs, masks, min_active=5, fixed=(0, D.N_DAYS - 1)) -> N.Network:
    runs = runs.assign(n_dates=D.popcount_rows(masks))
    cls = N.ride_classes(runs, 5, 0.05, True)
    return N.build_links(runs, masks, cls, HUBS, min_active, fixed)


def network(calls_df, journeys_df, masks, tol=2) -> tuple[N.Trips, N.Network]:
    """The build of pathset-network.py, steps b to h, on small tables."""
    trips = N.build_trips(calls_df, journeys_df, HUBS["tz_std"].to_numpy(), masks)
    legs, _ = N.build_legs(trips.calls, trips.trips["mode"].to_numpy(),
                           HUBS["fua"].to_numpy(), HUBS["tz_std"].to_numpy())
    runs, run_of_leg = N.merge_equal(legs)
    run_masks = N.or_calendars(run_of_leg, len(runs), legs["trip"],
                               legs["shift"], trips.masks)
    runs, run_masks, kept_of = N.merge_near(runs, run_masks, tol)
    net = links_of(runs, run_masks)
    event_of_leg = net.event_of_run[kept_of[run_of_leg]]
    net.events["trip"] = N.vehicle_keys(legs["trip"], event_of_leg,
                                        len(net.events))
    return trips, net


# --- a. Hubs -----------------------------------------------------------------
def test_hub_clustering_single_linkage_inside_one_fua():
    m = 1 / 111195                       # degrees of longitude per metre at lat 0
    stations = pd.DataFrame({
        "stn": [13, 12, 11, 10, 20],
        "name": ["Oost", "Centraal | Central", "Midden", "West", "Grens"],
        "fua": [1, 1, 1, 1, 2],
        "lon": [1100 * m, 600 * m, 300 * m, 0.0, 100 * m],
        "lat": 0.0,
        "calls": [5, 90, 10, 20, 7]})
    airports = pd.DataFrame({"iata": ["AAA"], "name": ["Alpha"], "icc": ["UK"],
                             "lon": [1.0], "lat": [1.0]})
    fuas = pd.DataFrame({"id": [1, 2], "icc": ["NL", "FI"]})
    nodes = pd.DataFrame({"node_id": ["FUA_001", "STN_0001", "AIR_AAA"],
                          "lon": [0.0, 350 * m, 1.0], "lat": [0.0, 0.0, 1.0]})
    hubs, stn2hub = N.build_hubs(stations, airports, fuas, nodes, 400, 400)

    # 10-11-12 chain at 300 m steps; 13 is 500 m further; 20 is 100 m from 10
    # but in another FUA
    assert stn2hub.to_dict() == {10: 0, 11: 0, 12: 0, 13: 1, 20: 2}
    assert list(hubs.columns) == N.S.HUBS
    assert hubs["kind"].tolist() == ["S", "S", "S", "A"]
    assert hubs["stns"].iloc[0].tolist() == [10, 11, 12]
    assert hubs["name"].tolist() == ["Centraal", "Oost", "Grens", "Alpha"]
    assert hubs["fua"].tolist() == [1, 1, 2, -1]
    assert hubs["tz_std"].tolist() == [1, 1, 2, 0]
    assert hubs["lon"].iloc[0] == pytest.approx(300 * m)
    # released station node: 50 m from station 11, 250 m from station 20,
    # 750 m from station 13
    assert hubs["released_id"].tolist() == ["STN_0001", "", "STN_0001", "AIR_AAA"]
    assert hubs["match_m"].iloc[0] == pytest.approx(50, abs=0.5)
    assert hubs["match_m"].iloc[2] == pytest.approx(250, abs=0.5)
    assert np.isnan(hubs["match_m"].iloc[1])
    assert hubs["iata"].tolist() == ["", "", "", "AAA"]


def test_single_linkage_radius_is_inclusive_of_chains_only():
    m = 1 / 111195
    lab = N.single_linkage([1, 1, 1], [0.0, 399 * m, 801 * m], [0.0] * 3, 400)
    assert lab.tolist() == [0, 0, 1]


def test_country_offsets_match_air_times():
    upstream = N.upstream_country_tz(
        Path(__file__).resolve().parent.parent / "air-times.py")
    assert upstream == N.TZ_STD
    assert {c for c, h in N.TZ_STD.items() if h == 0} == {"UK", "IE", "PT"}
    assert {c for c, h in N.TZ_STD.items() if h == 2} == {
        "FI", "EE", "LV", "LT", "RO", "BG", "EL"}
    with pytest.raises(KeyError):
        N.tz_std_of(["NL", "XX"])


# --- b. Trips ----------------------------------------------------------------
def test_local_clock_by_offset_and_summer_flag():
    assert N.local_clock(600, [0, 1, 2], False).tolist() == [600, 660, 720]
    assert N.local_clock(600, [0, 1, 2], True).tolist() == [660, 720, 780]


def test_summer_flag_needs_half_of_the_active_days():
    masks = cal(JULY[:4], WINTER[:4],
                list(WINTER[:2]) + list(JULY[:2]),
                list(WINTER[:3]) + list(JULY[:2]), [])
    assert N.summer_flags(masks).tolist() == [True, False, True, False, False]


def test_dateset_masks_checks_the_timeline():
    cols = D.date_columns()
    df = pd.DataFrame(np.zeros((2, len(cols)), dtype=bool), columns=cols)
    df.iloc[1, 3] = True
    df.insert(0, "dateset", [7, 9])
    ids, masks = N.dateset_masks(df)
    assert ids.tolist() == [7, 9] and days(masks[1]).tolist() == [3]
    with pytest.raises(ValueError):
        N.dateset_masks(df.drop(columns=cols[-1]))


def test_trip_times_are_local_and_ride_is_elapsed():
    # hub 0 (UTC+1) 08:20 UTC -> hub 2 (UTC+0) 10:00 UTC, winter and summer
    c = calls((1, [(0, 500, 500), (2, 600, 600)]),
              (2, [(0, 500, 500), (2, 600, 600)]))
    trips, net = network(c, journeys(2), cal(WINTER, JULY))
    assert len(trips.trips) == 2          # different local clocks
    ev = net.events.sort_values("dep")
    assert ev["dep"].tolist() == [560, 620]        # 09:20 CET, 10:20 CEST
    assert ev["ride"].tolist() == [100, 100]
    link = net.links.iloc[0]
    assert (link["src"], link["trg"], link["mode"]) == (0, 2, "R")
    assert link["tz_shift"] == -60
    # arrival clock at hub 2 = dep + ride + tz_shift = 10:00 GMT
    assert 560 + 100 + link["tz_shift"] == 600


def test_duplicated_feed_counts_once():
    # the same train in two feeds, calendars overlapping in part
    stops = [(0, 500, 500), (2, 600, 600)]
    c = calls((1, stops), (2, stops))
    trips, net = network(c, journeys(2), cal(WINTER[:30], WINTER[10:]))
    assert len(trips.trips) == 1
    assert trips.trips["n_jrn"].tolist() == [2]
    assert days(trips.masks[0]).tolist() == WINTER.tolist()
    assert len(net.events) == 1 and net.events["n_dates"].iloc[0] == len(WINTER)
    assert net.links["freq"].iloc[0] == 1
    assert net.links["freq_max"].iloc[0] == 1
    assert net.links["active_dates"].iloc[0] == len(WINTER)


def test_departure_after_midnight_is_normalised():
    # journey 1 leaves hub 0 at 24:10 local on Mondays (service day), journey
    # 2 at 00:10 on Tuesdays: one trip, on Tuesdays only
    c = calls((1, [(0, 1390, 1390), (2, 1490, 1490)]),
              (2, [(0, -50, -50), (2, 50, 50)]))
    trips, net = network(c, journeys(2), cal(MONDAYS, MONDAYS + 1))
    assert len(trips.trips) == 1
    assert trips.calls["dep"].iloc[0] == 10
    got = days(trips.masks[0])
    assert got.tolist() == (MONDAYS + 1).tolist()
    assert (WEEKDAY[got] == 1).all()
    assert net.events["dep"].tolist() == [10]
    assert net.events["n_dates"].tolist() == [len(MONDAYS)]
    assert net.links["freq"].iloc[0] == 1


def test_leg_after_midnight_departs_on_the_next_date():
    # 22:40 at hub 0 on Mondays, 24:40 at hub 4, on to hub 3 (UTC+2)
    c = calls((1, [(0, 1300, 1300), (4, 1400, 1420), (3, 1500, 1500)]))
    trips, net = network(c, journeys(1), cal(MONDAYS))
    ev = net.events.merge(net.links[["link", "src", "trg"]], on="link")
    ev["weekday"] = [WEEKDAY[days(m)].tolist() for m in net.masks]
    by = ev.set_index(["src", "trg"])
    assert by.loc[(0, 4), "dep"] == 1360 and set(by.loc[(0, 4), "weekday"]) == {0}
    assert by.loc[(4, 3), "dep"] == 40 and set(by.loc[(4, 3), "weekday"]) == {1}
    assert by.loc[(0, 3), "ride"] == 200 and set(by.loc[(0, 3), "weekday"]) == {0}
    # one vehicle: the same trip key on all its legs
    assert ev["trip"].nunique() == 1


def test_consecutive_calls_at_one_hub_are_one_call():
    c = calls((1, [(0, 500, 501), (0, 505, 510), (2, 600, 600)]))
    trips = N.build_trips(c, journeys(1), HUBS["tz_std"].to_numpy(), cal(WINTER))
    assert trips.n_calls_merged == 1
    assert trips.calls[["hub", "arr", "dep"]].values.tolist() == [
        [0, 560, 570], [2, 600, 600]]


def test_mode_of_a_trip_with_both_flags_follows_the_route_type():
    sig = np.array([0, 0, 0, 1, 1, 2, 2])
    rail = [True, False, False, True, True, True, False]
    route_type = [2, 3, 3, 2, 2, 2, 700]
    mode, mixed = N.signature_modes(sig, 3, rail, route_type)
    # 0: two journeys say bus; 1: rail only; 2: a tie goes to the extended type
    assert mode.tolist() == [B, R, B]
    assert mixed.tolist() == [True, False, True]


def test_signatures_separate_different_times_and_sequences():
    start = np.array([0, 2, 4, 6, 9])
    hub = np.array([0, 2, 0, 2, 0, 2, 0, 4, 2])
    arr = np.array([5, 9, 5, 9, 5, 8, 5, 7, 9])
    dep = np.array([5, 9, 5, 9, 5, 8, 5, 7, 9])
    assert N.signatures(start, hub, arr, dep).tolist() == [0, 0, 1, 2]


# --- c, d. Legs and hygiene ---------------------------------------------------
def test_expand_pairs_skips_one_fua_and_returns_positions():
    trip = np.array([7, 7, 7, 8, 8])
    hub = np.array([0, 1, 2, 2, 0])
    fua = np.array([10, 10, 11, 11, 10])
    arr = np.array([0, 10, 30, 100, 160])
    dep = np.array([1, 12, 30, 100, 160])
    t, src, trg, time, i, j, n_same = N.expand_pairs(trip, hub, fua, arr, dep)
    assert t.tolist() == [7, 7, 8]
    assert list(zip(src, trg)) == [(0, 2), (1, 2), (2, 0)]
    assert time.tolist() == [29, 18, 60]
    assert i.tolist() == [0, 1, 3] and j.tolist() == [2, 2, 4]
    assert n_same == 1


def test_hygiene_reasons():
    code = N.hygiene(ride=[0, 10, 1441, 60, 60, 60, 60],
                     km=[10, 10, 10, 400, 100, 140, 5000],
                     mode=[R, R, R, R, B, B, A],
                     max_ride=1440, v_max={"R": 350, "B": 130})
    assert [N.HYGIENE_REASONS[c] for c in code] == [
        "ride <= 0", "kept", "ride > MAX_RIDE_MIN", "speed above V_MAX_KMH",
        "kept", "speed above V_MAX_KMH", "kept"]


# --- e. Runs -----------------------------------------------------------------
def test_equal_runs_merge_and_keep_the_smallest_trip():
    legs = pd.DataFrame({
        "src": [0, 0, 0, 2], "trg": [2, 2, 2, 0], "mode": [R, R, B, R],
        "dep": [600, 600, 600, 600], "ride": [60, 60, 60, 60],
        "trip": [5, 3, 4, 6], "shift": [0, 0, 0, 1]})
    trip_masks = cal([], [], [], [3, 4], [9], [1, 2, 3], [7])
    runs, run_of_leg = N.merge_equal(legs)
    masks = N.or_calendars(run_of_leg, len(runs), legs["trip"], legs["shift"],
                           trip_masks)
    assert runs[["src", "trg", "mode", "trip"]].values.tolist() == [
        [0, 2, B, 4], [0, 2, R, 3], [2, 0, R, 6]]
    assert run_of_leg.tolist() == [1, 1, 0, 2]
    assert days(masks[1]).tolist() == [1, 2, 3, 4]
    assert days(masks[2]).tolist() == [8]          # shifted by one day


def test_near_duplicates_merge_into_the_run_with_more_dates():
    runs = runs_of([
        (0, 2, R, 600, 60),      # 4 dates
        (0, 2, R, 600, 63),      # ride 3 min apart: stays
        (0, 2, R, 601, 60),      # no common date: stays
        (0, 2, R, 602, 61),      # within 2 min, common date: absorbed by row 0
        (0, 2, B, 601, 60),      # other mode: stays
    ])
    masks = cal([1, 2, 3, 4], [1], [10], [4, 5], [1, 2])
    out, out_masks, kept_of = N.merge_near(runs, masks, 2)
    assert out[["dep", "ride", "trip"]].values.tolist() == [
        [600, 60, 0], [600, 63, 1], [601, 60, 2], [601, 60, 4]]
    assert kept_of.tolist() == [0, 1, 2, 0, 3]
    assert days(out_masks[0]).tolist() == [1, 2, 3, 4, 5]
    assert out["n_dates"].tolist() == [5, 1, 1, 2]


def test_near_duplicate_chains_do_not_drift():
    chain = [(0, 2, R, 600, 60), (0, 2, R, 602, 60), (0, 2, R, 604, 60)]
    # most dates at the start: 602 joins 600; 604 is 4 min from 600 and stays
    out, masks, kept_of = N.merge_near(
        runs_of(chain), cal(range(10), range(5), range(3)), 2)
    assert out["dep"].tolist() == [600, 604] and kept_of.tolist() == [0, 0, 1]
    assert out["n_dates"].tolist() == [10, 3]
    # most dates in the middle: both neighbours join it
    out, masks, kept_of = N.merge_near(
        runs_of(chain), cal(range(5), range(10), range(3)), 2)
    assert out["dep"].tolist() == [602] and kept_of.tolist() == [0, 0, 0]
    # a run that overlaps only the calendar grown by an earlier merge joins too
    out, masks, kept_of = N.merge_near(
        runs_of(chain[:2] + [(0, 2, R, 602, 62)]),
        cal(range(5), range(10), [3, 30]), 2)
    assert out["dep"].tolist() == [602] and days(masks[0])[-1] == 30


def test_near_merge_repeats_until_no_near_duplicate_is_left():
    # 602 (7 dates) is kept beside 600 (10 dates): no common date yet. 601
    # (2 dates) joins 600 and brings day 30, which 602 also has, so a single
    # pass would leave 600 and 602 as near-duplicates.
    runs = runs_of([(0, 2, R, 600, 60), (0, 2, R, 601, 60), (0, 2, R, 602, 60)])
    masks = cal(range(10), [5, 30], range(30, 37))
    out, out_masks, kept_of = N.merge_near(runs, masks, 2)
    assert out["dep"].tolist() == [600] and kept_of.tolist() == [0, 0, 0]
    assert days(out_masks[0]).tolist() == list(range(10)) + list(range(30, 37))
    assert out["n_dates"].tolist() == [17]
    # the same, but 602 has taken 604 in the first pass: joining 600 would
    # file 604 4 min from its own departure, so 602 stays
    runs = runs_of([(0, 2, R, 600, 60), (0, 2, R, 601, 60),
                    (0, 2, R, 602, 60), (0, 2, R, 604, 60)])
    masks = cal(range(10), [5, 30], range(30, 37), [31])
    out, out_masks, kept_of = N.merge_near(runs, masks, 2)
    assert out["dep"].tolist() == [600, 602]
    assert kept_of.tolist() == [0, 0, 1, 1]


def test_vehicle_keys_join_trips_that_share_an_event():
    # trips 4 and 7 share event 1; trip 7 also runs event 2; trip 2 runs
    # event 0 alone; trip 9's leg was not written (-1)
    key = N.vehicle_keys([4, 4, 7, 7, 2, 9], [0, 1, 1, 2, 3, -1], 4)
    assert key.tolist() == [4, 4, 4, 2]
    # a chain through a shared event joins all three trips
    assert N.vehicle_keys([5, 3, 3, 1], [0, 0, 1, 1], 2).tolist() == [1, 1]
    with pytest.raises(ValueError):
        N.vehicle_keys([1], [0], 2)                          # event 1 has no leg


def test_a_vehicle_keeps_one_key_through_a_shared_leg():
    # T1 0 -> 2 -> 3 and T2 0 -> 2 -> 4 share the leg 0 -> 2 (same times)
    c = calls((1, [(0, 500, 500), (2, 600, 610), (3, 700, 700)]),
              (2, [(0, 500, 500), (2, 600, 610), (4, 720, 720)]))
    _, net = network(c, journeys(2), cal(ALL_DAYS, ALL_DAYS))
    ev = net.events.merge(net.links[["link", "src", "trg"]], on="link")
    key = ev.set_index(["src", "trg"])["trip"]
    assert key.loc[(0, 2)] == key.loc[(2, 3)] == key.loc[(2, 4)]


# --- f. Flights --------------------------------------------------------------
def test_op_days_and_date_ranges():
    assert N.op_days_bits(["1234567", "1 3 5", "      7"]).tolist() == [127, 21, 64]
    day0, day1 = N.day_range(["2022-12-25", "2023-01-02", "2027-01-05"],
                             ["2023-01-03", "2023-01-15", "2027-01-09"])
    assert day0.tolist() == [0, 1, D.N_DAYS]
    assert day1.tolist() == [2, 14, D.N_DAYS - 1]


def test_weekday_calendars():
    # Mondays of 2-15 January 2023 and the Tuesday of 9-10 January
    masks = N.weekday_calendars([0, 0, 1], 2, [1, 8, 5], [14, 9, 4],
                                np.array([1, 2, 127], dtype=np.uint8))
    assert days(masks[0]).tolist() == [1, 8, 9]
    assert days(masks[1]).tolist() == []            # range clipped away


def test_air_alignment_fails_loudly():
    tt = pd.DataFrame({"src": ["AMS", "LHR"], "trg": ["LHR", "AMS"],
                       "carrier": ["KL", "BA"], "flight": [1, 1],
                       "start_date": ["2023-01-02"] * 2})
    N.check_air_alignment(tt, tt.assign(duration=[80, 75]))
    with pytest.raises(ValueError):
        N.check_air_alignment(tt, tt.iloc[::-1])
    with pytest.raises(ValueError):
        N.check_air_alignment(tt, tt.iloc[:1])
    with pytest.raises(ValueError, match="carrier"):
        N.check_air_alignment(tt, tt.assign(carrier=["BA", "KL"]))


# --- g, h. Classes, links, frequencies ----------------------------------------
def test_riding_time_classes():
    runs = runs_of([(0, 2, R, 600 + 10 * i, r)
                    for i, r in enumerate([70, 60, 64, 71, 62])]
                   + [(2, 0, R, 600, 71), (2, 0, R, 660, 100),
                      (0, 2, A, 600, 60), (0, 2, A, 700, 100)])
    # anchor 60 takes 60..65, anchor 70 takes 70..75 (max(5, 5 %) = 5)
    cls = N.ride_classes(runs, 5, 0.05, True)
    assert cls.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0]
    assert N.ride_classes(runs, 5, 0.05, False).tolist()[-2:] == [0, 1]
    # the relative width takes over above 100 min: 200 takes up to 210
    long = runs_of([(0, 2, R, 600, 200), (0, 2, R, 660, 210), (0, 2, R, 720, 211)])
    assert N.ride_classes(long, 5, 0.05, True).tolist() == [0, 0, 1]


def test_frequency_is_the_median_over_active_dates():
    mondays = ALL_DAYS[WEEKDAY == 0]
    even, odd = ALL_DAYS[::2], ALL_DAYS[1::2]
    runs = runs_of([
        (0, 2, R, 600, 60),                                   # weekly
        (0, 3, R, 600, 60), (0, 3, R, 700, 60),               # alternating days
        (0, 4, R, 500, 60), (0, 4, R, 600, 60), (0, 4, R, 700, 60),   # 3 a day
        (2, 0, R, 500, 60), (2, 0, R, 600, 60), (2, 0, R, 700, 60),   # 2 + Mondays
    ])
    masks = cal(mondays, even, odd, ALL_DAYS, ALL_DAYS, ALL_DAYS,
                ALL_DAYS, ALL_DAYS, mondays)
    net = links_of(runs, masks, fixed=(0, 99))
    l = net.links.set_index(["src", "trg"])
    assert l["freq"].tolist() == [1, 1, 3, 2]
    assert l["freq_max"].tolist() == [1, 1, 3, 3]
    assert l["active_dates"].tolist() == [len(mondays), D.N_DAYS, D.N_DAYS, D.N_DAYS]
    weekly = l.loc[(0, 2)]
    span = mondays[-1] - mondays[0] + 1
    assert (weekly["first_day"], weekly["last_day"]) == (mondays[0], mondays[-1])
    assert weekly["freq_span"] == pytest.approx(len(mondays) / span)
    assert weekly["active_share"] == pytest.approx(len(mondays) / span)
    assert weekly["freq_fixed"] == pytest.approx((mondays < 100).sum() / 100)
    assert l.loc[(0, 4), "freq_fixed"] == pytest.approx(3)
    assert l.loc[(0, 3), "freq_span"] == pytest.approx(1)


def test_a_duplicated_run_still_counts_once():
    legs = pd.DataFrame({"src": [0, 0], "trg": [2, 2], "mode": [R, R],
                         "dep": [600, 600], "ride": [60, 60],
                         "trip": [0, 1], "shift": [0, 0]})
    runs, run_of_leg = N.merge_equal(legs)
    masks = N.or_calendars(run_of_leg, len(runs), legs["trip"], legs["shift"],
                           cal(ALL_DAYS, ALL_DAYS))
    net = links_of(runs, masks)
    assert net.links["freq"].tolist() == [1]
    assert net.links["n_events"].tolist() == [1]


def test_link_attributes_and_event_order():
    runs = runs_of([(0, 2, R, 700, 62), (0, 2, R, 600, 60), (0, 2, R, 600, 58),
                    (0, 2, R, 800, 90)])
    masks = cal(range(10), range(30), range(40, 50), range(20))
    net = links_of(runs, masks)
    assert list(net.links.columns) == N.S.LINKS
    assert list(net.events.columns) == N.S.EVENTS
    assert net.links["cls"].tolist() == [0, 1]
    fast = net.links.iloc[0]
    assert (fast["ev0"], fast["ev1"], fast["n_events"]) == (0, 3, 3)
    assert (fast["ride_min"], fast["ride_max"]) == (58, 62)
    assert fast["ride"] == pytest.approx((62 * 10 + 60 * 30 + 58 * 10) / 50)
    assert fast["active_dates"] == 40
    assert fast["km"] == pytest.approx(
        float(N.gc_km(4.0, 52.0, 0.0, 51.5)), rel=1e-6)
    # events sorted by (link, dep, ride), calendars row-aligned
    assert net.events[["link", "dep", "ride"]].values.tolist() == [
        [0, 600, 58], [0, 600, 60], [0, 700, 62], [1, 800, 90]]
    assert net.events["n_dates"].tolist() == [10, 30, 10, 20]
    assert days(net.masks[0])[0] == 40
    assert net.event_of_run.tolist() == [2, 1, 0, 3]


def test_links_below_min_active_dates_go_to_sparse():
    runs = runs_of([(0, 2, R, 600, 60), (0, 2, R, 600, 120), (0, 3, R, 600, 60)])
    net = links_of(runs, cal(range(5), range(4), range(4)), min_active=5)
    assert len(net.links) == 1 and net.links["active_dates"].tolist() == [5]
    assert net.links[["link", "ev0", "ev1"]].values.tolist() == [[0, 0, 1]]
    assert list(net.sparse.columns) == [
        c for c in N.S.LINKS if c not in ("ev0", "ev1")]
    assert net.sparse[["link", "src", "trg", "cls"]].values.tolist() == [
        [0, 0, 2, 1], [1, 0, 3, 0]]
    assert len(net.events) == 1 and len(net.masks) == 1
    assert net.event_of_run.tolist() == [0, -1, -1]


def test_link_members_count_events_once():
    members = N.link_members(
        event=[0, 0, 1, 1, 2, 2], link_of_event=[0, 0, 1],
        agency=["X", "X", "X", "Y", "Y", "KL"], route_type=[2, 2, 2, 2, 3, -1],
        line=[7, 7, 7, 8, 9, -1])
    assert list(members.columns) == N.S.LINK_MEMBERS
    assert members.values.tolist() == [
        [0, "X", 2, 7, 2], [0, "Y", 2, 8, 1], [1, "KL", -1, -1, 1],
        [1, "Y", 3, 9, 1]]


# --- i. Car ------------------------------------------------------------------
CROSSING = (("Medway", "Dunkerque", 178, 164, 183, 161),)
NAMES = pd.Series(["Xa", "Medway", "Dunkerque", "Yb"])


def car_table(crossing_min_factor: float, crossing_km_factor: float) -> pd.DataFrame:
    """Two road blocks {0, 1} and {2, 3} joined between FUA 1 and FUA 2 the
    way car-times.py joins them; the factors say how much of the crossing
    constants the cross-block rows carry."""
    t = {(0, 1): 30, (1, 0): 32, (2, 3): 50, (3, 2): 52}
    k = {(0, 1): 40, (1, 0): 41, (2, 3): 60, (3, 2): 61}
    for i in range(4):
        t[i, i] = k[i, i] = 0
    rows = [(a, b, t[a, b], k[a, b]) for a, b in t]
    for y in (0, 1):
        for x in (2, 3):
            rows.append((y, x, t[y, 1] + t[2, x] + 164 * crossing_min_factor,
                         k[y, 1] + k[2, x] + 178 * crossing_km_factor))
            rows.append((x, y, t[x, 2] + t[1, y] + 161 * crossing_min_factor,
                         k[x, 2] + k[1, y] + 183 * crossing_km_factor))
    return pd.DataFrame(rows, columns=["src_fua", "trg_fua", "time", "dist"]
                        ).astype({"time": "float32", "dist": "float32"})


def test_car_crossing_is_restored():
    car = N.build_car(car_table(1 / 60, 1 / 1000), NAMES, CROSSING)
    assert list(car.columns) == N.S.CAR
    assert len(car) == 12                           # no diagonal
    row = car.set_index(["o", "d"])
    assert row.loc[(0, 3), "minutes"] == pytest.approx(30 + 164 + 50, abs=1e-3)
    assert row.loc[(0, 3), "km"] == pytest.approx(40 + 178 + 60, abs=1e-3)
    assert row.loc[(3, 0), "minutes"] == pytest.approx(52 + 161 + 32, abs=1e-3)
    assert row.loc[(2, 1), "km"] == pytest.approx(183, abs=1e-3)
    assert row.loc[(0, 1), "minutes"] == 30
    assert row["fixed"].sum() == 8
    assert not row.loc[(0, 1), "fixed"] and row.loc[(1, 2), "fixed"]

def test_crossing_sides_are_read_from_the_table():
    t = car_table(1 / 60, 1 / 1000).set_index(["src_fua", "trg_fua"])["time"]
    side_a, side_b = N.crossing_sides(t.astype(float), 1, 2, 164)
    assert sorted(side_a) == [0, 1] and sorted(side_b) == [2, 3]


def test_car_table_without_the_unit_bug_is_refused():
    with pytest.raises(ValueError):
        N.build_car(car_table(1, 1), NAMES, CROSSING)


def test_car_speed_on_the_straight_line():
    car = pd.DataFrame({"o": [0], "d": [1], "minutes": [30.0]})
    kmh = N.car_speed(car, fua=[0, 1], lon=[0.0, 1.0], lat=[0.0, 0.0])
    assert kmh[0] == pytest.approx(2 * 111.195, rel=1e-4)
