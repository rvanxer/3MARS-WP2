"""Generalised travel time: hand-computed paths, monotonicity, hidden wait,
agreement of the table form and the kernels, lower bounds and the capped
frequency they need, parameters read at call time, no disk cache on compiled
callers of the kernels."""

import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import cost as C  # noqa: E402
from pathset import io  # noqa: E402
from pathset.schemas import MODES  # noqa: E402

A, B, R = C.MODE_CODE["A"], C.MODE_CODE["B"], C.MODE_CODE["R"]

# Deck slide 229, with the placeholders of the plan for the hidden wait and
# the car. The hand computations below use these numbers, not the YAML file.
DECK = {
    "BETA_ACCESS": 1.87, "BETA_EGRESS": 1.87,
    "BETA_IVT": {"A": 1.56, "R": 0.96, "B": 0.88, "C": 1.0},
    "BETA_TRANS": 1.76, "TAU_TP": 10,
    "TAU_DEP": {"A": 90, "R": 15, "B": 15},
    "TAU_ARR": {"A": 45, "R": 5, "B": 5},
    "BETA_HWT": 600, "HWT_MAX": 240,
}
PAR = C.param_array(DECK)
NAN = float("nan")
KEYS = C.COMPONENTS + ("T", "gtt", "hwt", "gtc")


def one(n, acc, modes, rides, fts, egr, freq, par=PAR):
    """Table form for one path, as a dict of floats."""
    out = C.path_costs([n], [acc], [modes], [rides], [fts], [egr], [freq], par)
    return {k: float(v[0]) for k, v in out.items()}


def kernel(n, acc, modes, rides, fts, egr, freq, par=PAR):
    """Kernel form for one path, as a dict with the same keys."""
    args = (n, acc, *modes, *rides, *fts, egr)
    out = dict(zip(C.COMPONENTS, C.k_components(*args, par)))
    out["T"], out["gtt"], out["hwt"], out["gtc"] = C.k_costs(*args, freq, par)
    assert out["T"] == C.k_clock(*args, par)
    assert out["gtt"] == C.k_gtt(*args, par)
    return out


def random_table(seed, size):
    """Paths with 1..3 links; slots beyond the link count hold rubbish."""
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 4, size)
    used = np.arange(3) < n[:, None]
    modes = np.where(used, rng.integers(0, 3, (size, 3)), -1)
    rides = np.where(used, rng.uniform(10, 600, (size, 3)), np.nan)
    fts = np.where(used[:, 1:], rng.uniform(5, 300, (size, 2)), np.nan)
    acc = rng.uniform(0, 120, size)
    egr = rng.uniform(0, 120, size)
    freq = rng.uniform(0.2, 40, size)
    return n, acc, modes, rides, fts, egr, freq


# --- Parameters --------------------------------------------------------------
def test_mode_codes_follow_the_contract():
    assert C.MODE_CODE == {"A": 0, "B": 1, "R": 2, "C": 3}
    assert [C.MODE_CODE[m] for m in MODES] == [0, 1, 2]


def test_param_array_layout():
    assert PAR.shape == (C.N_PAR,) and PAR.dtype == np.float64
    assert list(PAR[C.IVT0:C.IVT0 + 4]) == [1.56, 0.88, 0.96, 1.0]   # A, B, R, C
    assert list(PAR[C.DEP0:C.DEP0 + 4]) == [90, 15, 15, 0]           # car: none
    assert list(PAR[C.ARR0:C.ARR0 + 4]) == [45, 5, 5, 0]
    assert PAR[C.I_ACCESS] == 1.87 and PAR[C.I_EGRESS] == 1.87
    assert PAR[C.I_TRANS] == 1.76 and PAR[C.I_TP] == 10
    assert PAR[C.I_HWT] == 600 and PAR[C.I_HWT_MAX] == 240


def test_param_array_reads_the_parameter_file():
    par = C.param_array(io.P)
    assert np.isfinite(par).all()
    assert par[C.I_TRANS] == io.P["BETA_TRANS"]
    assert par[C.IVT0 + A] == io.P["BETA_IVT"]["A"]
    assert par[C.DEP0 + R] == io.P["TAU_DEP"]["R"]
    assert par[C.ARR0 + B] == io.P["TAU_ARR"]["B"]
    assert par[C.I_HWT_MAX] == io.P["HWT_MAX"]


# --- Hand-computed paths (like the Hague-Paris example, deck slide 228) ------
def test_rail_rail_rail_by_hand():
    path = (3, 10, (R, R, R), (25, 60, 120), (12, 20), 15, 10)
    # riding 25 + 60 + 120 = 205; transfers 12 + 20 = 32
    # T       = 10 + 15 + 205 + 32 + 5 + 15                     = 282
    # t_a     = 1.87 * 10                                       = 18.7
    # t_dep   = 15 (rail)
    # t_ivt   = 0.96 * 205                                      = 196.8
    # t_trans = 1.76 * 32                                       = 56.32
    # t_tp    = 10 * 2                                          = 20
    # t_arr   = 5 (rail)
    # t_e     = 1.87 * 15                                       = 28.05
    # gtt     = 18.7 + 15 + 196.8 + 56.32 + 20 + 5 + 28.05      = 339.87
    # hwt     = 600 / (2 * 10)                                  = 30
    # gtc     = 339.87 + 30                                     = 369.87
    want = {"t_a": 18.7, "t_dep": 15, "t_ivt": 196.8, "t_trans": 56.32,
            "t_tp": 20, "t_arr": 5, "t_e": 28.05, "T": 282, "gtt": 339.87,
            "hwt": 30, "gtc": 369.87}
    for got in (one(*path), kernel(*path)):
        assert set(got) == set(want)
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=1e-9), k


def test_air_direct_by_hand():
    path = (1, 30, (A, -1, -1), (70, NAN, NAN), (NAN, NAN), 40, 4)
    # T       = 30 + 90 + 70 + 0 + 45 + 40                      = 275
    # t_a     = 1.87 * 30                                       = 56.1
    # t_dep   = 90 (air)
    # t_ivt   = 1.56 * 70                                       = 109.2
    # t_trans = 0, t_tp = 0 (no transfer)
    # t_arr   = 45 (air)
    # t_e     = 1.87 * 40                                       = 74.8
    # gtt     = 56.1 + 90 + 109.2 + 0 + 0 + 45 + 74.8           = 375.1
    # hwt     = 600 / (2 * 4)                                   = 75
    # gtc     = 375.1 + 75                                      = 450.1
    want = {"t_a": 56.1, "t_dep": 90, "t_ivt": 109.2, "t_trans": 0, "t_tp": 0,
            "t_arr": 45, "t_e": 74.8, "T": 275, "gtt": 375.1, "hwt": 75,
            "gtc": 450.1}
    for got in (one(*path), kernel(*path)):
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=1e-9), k


def test_coach_then_air_takes_first_and_last_allowance():
    path = (2, 20, (B, A, -1), (100, 80, NAN), (120, NAN), 25, 2)
    # T       = 20 + 15 + (100 + 80) + 120 + 45 + 25            = 405
    # t_a     = 1.87 * 20                                       = 37.4
    # t_dep   = 15 (coach, the first link)
    # t_ivt   = 0.88 * 100 + 1.56 * 80 = 88 + 124.8             = 212.8
    # t_trans = 1.76 * 120                                      = 211.2
    # t_tp    = 10 * 1                                          = 10
    # t_arr   = 45 (air, the last link)
    # t_e     = 1.87 * 25                                       = 46.75
    # gtt     = 37.4 + 15 + 212.8 + 211.2 + 10 + 45 + 46.75     = 578.15
    # hwt     = 600 / (2 * 2)                                   = 150
    want = {"t_a": 37.4, "t_dep": 15, "t_ivt": 212.8, "t_trans": 211.2,
            "t_tp": 10, "t_arr": 45, "t_e": 46.75, "T": 405, "gtt": 578.15,
            "hwt": 150, "gtc": 728.15}
    for got in (one(*path), kernel(*path)):
        for k in want:
            assert got[k] == pytest.approx(want[k], abs=1e-9), k


def test_car_by_hand():
    # T = 300; gtt = 1.0 * 300 = 300; no allowances; hwt = 0; gtc = 300
    got = C.car_costs([300.0], PAR)
    assert set(got) == set(KEYS)
    for k in ("T", "gtt", "gtc", "t_ivt"):
        assert got[k][0] == 300.0, k
    for k in ("t_a", "t_dep", "t_trans", "t_tp", "t_arr", "t_e", "hwt"):
        assert got[k][0] == 0.0, k
    assert C.k_car_gtt(300.0, PAR) == 300.0
    # with a weight of 1.2: gtt = 1.2 * 300 = 360, T stays 300
    heavy = copy.deepcopy(DECK)
    heavy["BETA_IVT"]["C"] = 1.2
    par = C.param_array(heavy)
    assert C.car_costs([300.0], par)["gtt"][0] == pytest.approx(360.0)
    assert C.car_costs([300.0], par)["T"][0] == 300.0
    assert C.k_car_gtt(300.0, par) == pytest.approx(360.0)


def test_gtt_is_the_sum_of_its_components():
    table = random_table(11, 500)
    out = C.path_costs(*table, PAR)
    total = sum(out[k] for k in C.COMPONENTS)
    np.testing.assert_allclose(out["gtt"], total, rtol=1e-12)
    np.testing.assert_allclose(out["gtc"], out["gtt"] + out["hwt"], rtol=1e-12)


# --- Hidden wait -------------------------------------------------------------
def test_hidden_wait_values_cap_zero_nan():
    # 600 / (2 * 10) = 30; 600 / (2 * 2) = 150; 600 / (2 * 1.25) = 240 (at the
    # cap); 600 / (2 * 1) = 300 -> 240; 600 / (2 * 0.1) = 3000 -> 240;
    # zero, negative and NaN -> 240; infinitely frequent -> 0
    freq = [10, 2, 1.25, 1, 0.1, 0, -3, NAN, np.inf]
    want = [30, 150, 240, 240, 240, 240, 240, 240, 0]
    np.testing.assert_allclose(C.hidden_wait(freq, PAR), want, rtol=1e-12)
    for f, w in zip(freq, want):
        assert C.k_hidden_wait(f, PAR) == pytest.approx(w, abs=1e-12)
    assert C.k_hidden_wait(3, PAR) == pytest.approx(100.0)   # integer count


def test_hidden_wait_falls_with_frequency():
    freq = np.linspace(0.05, 60, 400)
    hwt = C.hidden_wait(freq, PAR)
    assert (np.diff(hwt) <= 0).all()
    assert hwt[0] == 240 and hwt[-1] == pytest.approx(5.0)
    uncapped = hwt < 240
    assert (np.diff(hwt[uncapped]) < 0).all()


# --- Monotonicity ------------------------------------------------------------
def test_monotone_in_every_time_input():
    n, acc, modes, rides, fts, egr, freq = random_table(5, 400)
    base = C.path_costs(n, acc, modes, rides, fts, egr, freq, PAR)
    inputs = dict(acc=acc, rides=rides, fts=fts, egr=egr)

    def bumped(name, arr):
        a = {**inputs, name: arr}
        return C.path_costs(n, a["acc"], modes, a["rides"], a["fts"], a["egr"],
                            freq, PAR)

    for name in ("acc", "egr"):
        up = bumped(name, inputs[name] + 1.0)
        for k in ("T", "gtt", "gtc"):
            assert (up[k] > base[k]).all(), (name, k)
    # ride j belongs to link j; transfer j lies before link j + 1
    for name, slots, link_offset in (("rides", 3, 0), ("fts", 2, 1)):
        for j in range(slots):
            arr = inputs[name].copy()
            arr[:, j] += 1.0
            up = bumped(name, arr)
            in_use = n > j + link_offset
            for k in ("T", "gtt", "gtc"):
                assert (up[k][in_use] > base[k][in_use]).all(), (name, j, k)
                assert (up[k][~in_use] == base[k][~in_use]).all(), (name, j, k)


def test_monotone_in_frequency_and_frequency_is_not_in_T():
    n, acc, modes, rides, fts, egr, freq = random_table(6, 400)
    base = C.path_costs(n, acc, modes, rides, fts, egr, freq, PAR)
    more = C.path_costs(n, acc, modes, rides, fts, egr, freq * 2, PAR)
    assert (more["hwt"] <= base["hwt"]).all()
    assert (more["gtc"] <= base["gtc"]).all()
    assert (more["gtc"] < base["gtc"])[base["hwt"] < 240].all()
    assert (more["T"] == base["T"]).all()
    assert (more["gtt"] == base["gtt"]).all()


def test_a_transfer_adds_its_penalty_to_gtt_only():
    # the same riding time as one link or as two with a transfer of 0 minutes:
    # T is equal, gtt differs by TAU_TP = 10
    direct = one(1, 10, (R, -1, -1), (180, NAN, NAN), (NAN, NAN), 15, 5)
    split = one(2, 10, (R, R, -1), (60, 120, NAN), (0, NAN), 15, 5)
    assert split["T"] == pytest.approx(direct["T"])
    assert split["gtt"] - direct["gtt"] == pytest.approx(10.0)


# --- Table form against kernels ----------------------------------------------
def test_table_and_kernels_agree_on_random_paths():
    n, acc, modes, rides, fts, egr, freq = random_table(2026, 3000)
    freq[::7] = 0.0
    freq[3::11] = np.nan
    for par in (PAR, C.param_array(io.P)):
        table = C.path_costs(n, acc, modes, rides, fts, egr, freq, par)
        for i in range(n.size):
            args = (int(n[i]), acc[i], *modes[i], *rides[i], *fts[i], egr[i])
            comp = C.k_components(*args, par)
            total = C.k_costs(*args, freq[i], par)
            got = np.array(comp + total)
            want = np.array([table[k][i] for k in KEYS])
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9)


def test_unused_slots_are_ignored():
    clean = one(1, 30, (A, 0, 0), (70, 0, 0), (0, 0), 40, 4)
    dirty = one(1, 30, (A, -1, 7), (70, 999, NAN), (NAN, 500), 40, 4)
    assert clean == dirty
    assert kernel(1, 30, (A, -1, 7), (70, 999, NAN), (NAN, 500), 40, 4) \
        == kernel(1, 30, (A, 0, 0), (70, 0, 0), (0, 0), 40, 4)
    two = one(2, 30, (R, B, 0), (70, 50, 0), (15, 0), 40, 4)
    assert two == one(2, 30, (R, B, -5), (70, 50, NAN), (15, NAN), 40, 4)


def test_car_table_and_kernel_agree():
    minutes = np.random.default_rng(4).uniform(30, 1400, 200)
    table = C.car_costs(minutes, PAR)
    for i in range(minutes.size):
        assert C.k_car_gtt(minutes[i], PAR) == pytest.approx(table["gtt"][i], rel=1e-12)
    assert (table["T"] == minutes).all() and (table["hwt"] == 0).all()


# --- Lower bounds ------------------------------------------------------------
def test_sparsest_link():
    n = [1, 2, 3, 3]
    counts = [[4, 1, 1], [9, 6, 2], [9, 6, 2], [3, 30, 12]]
    assert list(C.sparsest(n, counts)) == [4, 6, 2, 3]
    for ni, c in zip(n, counts):
        assert C.k_sparsest(ni, *c) == min(c[:ni])


def test_bounds_by_hand():
    # rail-rail-rail of the first test with minimum transfers 5 and 8 and
    # largest daily counts 30, 12, 20 (sparsest 12):
    # T   = 10 + 15 + 205 + 13 + 5 + 15                         = 263
    # gtt = 18.7 + 15 + 196.8 + 1.76 * 13 + 20 + 5 + 28.05      = 306.43
    # hwt = 600 / (2 * 12)                                      = 25
    low = C.path_bounds([3], [10], [[R, R, R]], [[25, 60, 120]], [[5, 8]],
                        [15], [[30, 12, 20]], PAR)
    assert set(low) == {"T", "gtt", "hwt", "gtc"}
    want = {"T": 263, "gtt": 306.43, "hwt": 25, "gtc": 331.43}
    got = C.k_bounds(3, 10, R, R, R, 25, 60, 120, 5, 8, 15, 30, 12, 20, PAR)
    for k, g in zip(("T", "gtt", "hwt", "gtc"), got):
        assert low[k][0] == pytest.approx(want[k], abs=1e-9), k
        assert g == pytest.approx(want[k], abs=1e-9), k


def test_bounds_never_exceed_the_true_value():
    n, acc, modes, rides, fts, egr, _ = random_table(8, 2000)
    rng = np.random.default_rng(9)
    ft_min = fts * rng.uniform(0, 1, fts.shape)
    freq_max = rng.integers(1, 40, (n.size, 3))
    freq = C.sparsest(n, freq_max) * rng.uniform(0.02, 1, n.size)
    true = C.path_costs(n, acc, modes, rides, fts, egr, freq, PAR)
    low = C.path_bounds(n, acc, modes, rides, ft_min, egr, freq_max, PAR)
    for k in ("T", "gtt", "hwt", "gtc"):
        assert (low[k] <= true[k] + 1e-9).all(), k
    for i in range(0, n.size, 20):
        got = C.k_bounds(int(n[i]), acc[i], *modes[i], *rides[i], *ft_min[i],
                         egr[i], *freq_max[i], PAR)
        want = [low[k][i] for k in ("T", "gtt", "hwt", "gtc")]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9)


def test_capped_frequency():
    # counted frequency above, below and at the sparsest count; slots beyond
    # the link count are ignored; NaN and zero stay as they are
    n = [2, 2, 2, 1, 3, 2, 2]
    counts = [[20, 1, 9], [20, 1, 9], [20, 4, 1], [3, 1, 1], [9, 6, 2],
              [5, 5, 0], [5, 5, 0]]
    freq = [2, 0.5, 4, 7, 2.5, NAN, 0]
    want = [1, 0.5, 4, 3, 2, NAN, 0]
    np.testing.assert_array_equal(C.capped_frequency(n, freq, counts), want)
    for ni, f, c, w in zip(n, freq, counts, want):
        np.testing.assert_array_equal(C.k_capped_frequency(ni, f, *c), w)
    assert C.k_capped_frequency(2, 7, 3, 5, 0) == 3.0        # integer inputs
    with pytest.raises(ValueError):
        C.capped_frequency([1, 2], [1.0], [[1, 1, 1], [1, 1, 1]])


def test_hidden_wait_bound_needs_the_capped_frequency_by_hand():
    # rail-rail, access 10, rides 60 and 90, transfer 10 (also the minimum),
    # egress 15; largest daily counts 20 and 1; counted twice a day (two
    # departures of the daily link on a typical day that never share a date)
    # T    = 10 + 15 + 150 + 10 + 5 + 15                            = 205
    # gtt  = 18.7 + 15 + 0.96 * 150 + 1.76 * 10 + 10 + 5 + 28.05    = 238.35
    # bound:    hwt = min(600 / (2 * 1), 240) = 240, gtc            = 478.35
    # raw 2:    hwt = 600 / (2 * 2)           = 150, gtc            = 388.35
    #           90 minutes below the "lower bound"
    # capped 1: hwt = 240,                           gtc            = 478.35
    chain = (2, 10, R, R, -1, 60, 90, NAN, 10, NAN, 15)
    low = C.k_bounds(*chain, 20, 1, 0, PAR)
    assert low == pytest.approx((205, 238.35, 240, 478.35), abs=1e-9)
    raw = C.k_costs(*chain, 2, PAR)
    assert raw == pytest.approx((205, 238.35, 150, 388.35), abs=1e-9)
    assert not C.k_bounds_hold(low, raw)
    capped = C.k_costs(*chain, C.k_capped_frequency(2, 2, 20, 1, 0), PAR)
    assert capped == pytest.approx((205, 238.35, 240, 478.35), abs=1e-9)
    assert C.k_bounds_hold(low, capped)

    table = ([2], [10], [[R, R, -1]], [[60, 90, NAN]], [[10, NAN]], [15])
    counts = [[20, 1, 0]]
    low_t = C.path_bounds(*table, counts, PAR)
    assert not C.bounds_hold(low_t, C.path_costs(*table, [2], PAR))[0]
    assert C.bounds_hold(low_t, C.path_costs(
        *table, C.capped_frequency([2], [2], counts), PAR))[0]


def test_bounds_hold_for_any_counted_frequency_after_the_cap():
    n, acc, modes, rides, fts, egr, _ = random_table(21, 3000)
    rng = np.random.default_rng(22)
    ft_min = fts * rng.uniform(0, 1, fts.shape)
    freq_max = rng.integers(1, 40, (n.size, 3))
    cap = C.sparsest(n, freq_max)
    counted = cap * rng.uniform(0.02, 4, n.size)     # up to four times the cap
    counted[::19] = np.nan
    counted[7::23] = 0.0
    low = C.path_bounds(n, acc, modes, rides, ft_min, egr, freq_max, PAR)

    raw = C.path_costs(n, acc, modes, rides, fts, egr, counted, PAR)
    ok = C.bounds_hold(low, raw)
    assert ok[~(counted > cap)].all()                # within the cap: holds
    assert (~ok).sum() > 1000                        # above it: it does not
    assert (raw["T"] >= low["T"]).all() and (raw["gtt"] >= low["gtt"]).all()

    freq = C.capped_frequency(n, counted, freq_max)
    true = C.path_costs(n, acc, modes, rides, fts, egr, freq, PAR)
    assert C.bounds_hold(low, true).all()
    for i in range(0, n.size, 15):
        args = (int(n[i]), acc[i], *modes[i], *rides[i])
        k_low = C.k_bounds(*args, *ft_min[i], egr[i], *freq_max[i], PAR)
        k_freq = C.k_capped_frequency(int(n[i]), counted[i], *freq_max[i])
        np.testing.assert_array_equal(k_freq, freq[i])
        assert C.k_bounds_hold(k_low, C.k_costs(*args, *fts[i], egr[i], k_freq, PAR))
        assert C.k_bounds_hold(k_low, C.k_costs(*args, *fts[i], egr[i],
                                                counted[i], PAR)) == ok[i]


def test_bounds_hold_fails_on_nan_and_on_each_quantity():
    low = {k: np.array([1.0, 1.0, 1.0, 1.0, 1.0, NAN]) for k in C.BOUNDED}
    true = {k: np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0]) for k in C.BOUNDED}
    for row, k in enumerate(C.BOUNDED, start=1):      # one quantity too low
        true[k] = true[k].copy()
        true[k][row] = 0.5
    true["T"][0] = 1.0                                # equal: holds
    assert C.bounds_hold(low, true).tolist() == [True] + [False] * 5
    assert C.k_bounds_hold((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0))
    for j in range(4):
        true_j = [2.0] * 4
        true_j[j] = 0.5
        assert not C.k_bounds_hold((1.0, 1.0, 1.0, 1.0), tuple(true_j))
        true_j[j] = NAN
        assert not C.k_bounds_hold((1.0, 1.0, 1.0, 1.0), tuple(true_j))


# --- Compiled callers of the kernels must not be cached ----------------------
REPO = Path(__file__).resolve().parent.parent
JIT_NAMES = {"njit", "jit"}


def jit_flags(fn) -> tuple:
    """(compiled, cached) of a function definition, from its decorators.
    ``cache=`` anything but a literal False counts as cached."""
    compiled = cached = False
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) \
            else getattr(target, "id", "")
        if name not in JIT_NAMES:
            continue
        compiled = True
        for kw in getattr(dec, "keywords", []):
            if kw.arg == "cache" and not (isinstance(kw.value, ast.Constant)
                                          and kw.value.value is False):
                cached = True
    return compiled, cached


def cached_callers(sources: dict, kernels: set) -> list:
    """(file, line, function) of every compiled function with a disk cache
    that reaches one of ``kernels``, directly or through other compiled
    functions (rule of the pathset.cost docstring). Names are matched across
    files without resolving imports, which errs on the side of reporting."""
    compiled = []
    for path, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FunctionDef):
                continue
            is_jit, cached = jit_flags(node)
            if is_jit:
                used = {x.id if isinstance(x, ast.Name) else x.attr
                        for stmt in node.body for x in ast.walk(stmt)
                        if isinstance(x, (ast.Name, ast.Attribute))}
                compiled.append((path, node.lineno, node.name, cached, used))
    reach = set(kernels)
    grown = True
    while grown:
        grown = False
        for _, _, name, _, used in compiled:
            if name not in reach and used & reach:
                reach.add(name)
                grown = True
    return [(path, line, name) for path, line, name, cached, used in compiled
            if cached and used & reach]


def cost_kernels() -> set:
    return {name for name, obj in vars(C).items() if isinstance(obj, Dispatcher)}


def test_the_scan_finds_cached_callers():
    sources = {
        "direct.py": "from numba import njit\n"
                     "from pathset import cost as C\n"
                     "@njit(cache=True)\n"
                     "def f(par):\n"
                     "    return C.k_gtt(1, 0, 2, 0, 0, 9, 0, 0, 0, 0, 0, par)\n",
        "helper.py": "from numba import njit\n"
                     "from pathset.cost import k_hidden_wait\n"
                     "@njit\n"
                     "def inner(f, par):\n"
                     "    return k_hidden_wait(f, par)\n",
        "outer.py": "import numba\n"
                    "from helper import inner\n"
                    "@numba.njit(cache=True, parallel=True)\n"
                    "def g(f, par):\n"
                    "    return inner(f, par)\n"
                    "@numba.njit(cache=False)\n"
                    "def g_fresh(f, par):\n"
                    "    return inner(f, par)\n",
        "clean.py": "from numba import njit\n"
                    "@njit(cache=True)\n"
                    "def h(x):\n"
                    "    return x + 1\n"
                    "def plain(par):\n"
                    "    return k_gtt(par)\n",
    }
    assert sorted(cached_callers(sources, cost_kernels())) == [
        ("direct.py", 4, "f"), ("outer.py", 4, "g")]


def test_no_compiled_caller_of_the_kernels_is_cached():
    kernels = cost_kernels()
    assert {"k_gtt", "k_costs", "k_bounds", "k_hidden_wait", "_k_sums",
            "k_capped_frequency", "k_bounds_hold"} <= kernels
    files = [p for pattern in ("pathset/*.py", "pathset-*.py", "tests/*.py")
             for p in REPO.glob(pattern) if p != REPO / "pathset" / "cost.py"]
    assert REPO / "tests" / "test_cost_review.py" in files
    sources = {str(p.relative_to(REPO)): p.read_text(encoding="utf-8") for p in files}
    assert cached_callers(sources, kernels) == [], \
        "cache=True on a function that reaches a pathset.cost kernel: it " \
        "would keep the old cost formula after cost.py is edited"


# --- Parameters are read at call time ----------------------------------------
PATHS = [
    (3, 10, (R, R, R), (25, 60, 120), (12, 20), 15, 10),
    (1, 30, (A, -1, -1), (70, NAN, NAN), (NAN, NAN), 40, 4),
    (2, 20, (B, A, -1), (100, 80, NAN), (120, NAN), 25, 0.5),
    (2, 20, (A, B, -1), (80, 100, NAN), (90, NAN), 25, 3),
]


def all_results(params):
    """Every output of both forms for the example paths and a car trip."""
    par = C.param_array(params)
    out = []
    for path in PATHS:
        out += [one(*path, par=par)[k] for k in KEYS]
        out += [kernel(*path, par=par)[k] for k in KEYS]
    out += [C.car_costs([300.0], par)["gtt"][0], C.k_car_gtt(300.0, par)]
    return np.array(out)


@pytest.mark.parametrize("key, sub", [
    ("BETA_ACCESS", None), ("BETA_EGRESS", None), ("BETA_TRANS", None),
    ("TAU_TP", None), ("BETA_HWT", None), ("HWT_MAX", None),
    ("BETA_IVT", "A"), ("BETA_IVT", "B"), ("BETA_IVT", "R"), ("BETA_IVT", "C"),
    ("TAU_DEP", "A"), ("TAU_DEP", "B"), ("TAU_DEP", "R"),
    ("TAU_ARR", "A"), ("TAU_ARR", "B"), ("TAU_ARR", "R"),
])
def test_changing_a_parameter_changes_the_result(key, sub):
    before = all_results(DECK)
    changed = copy.deepcopy(DECK)
    if sub is None:
        changed[key] = changed[key] * 1.5
    else:
        changed[key][sub] = changed[key][sub] * 1.5
    after = all_results(changed)
    assert (after != before).any(), (key, sub)
    np.testing.assert_array_equal(all_results(DECK), before)


def test_changed_parameters_by_hand():
    changed = copy.deepcopy(DECK)
    changed["BETA_TRANS"] = 2.76      # + 1 per transfer minute
    changed["TAU_TP"] = 25            # + 15 per transfer
    par = C.param_array(changed)
    # rail-rail-rail: gtt = 339.87 + 1 * 32 + 15 * 2 = 401.87; T stays 282
    for got in (one(*PATHS[0], par=par), kernel(*PATHS[0], par=par)):
        assert got["gtt"] == pytest.approx(401.87, abs=1e-9)
        assert got["T"] == pytest.approx(282, abs=1e-9)
    # a copy of the parameter file works the same way and leaves io.P alone
    mine = copy.deepcopy(io.P)
    mine["BETA_ACCESS"] = io.P["BETA_ACCESS"] + 1.0
    d = one(*PATHS[0], par=C.param_array(mine))["gtt"] \
        - one(*PATHS[0], par=C.param_array(io.P))["gtt"]
    assert d == pytest.approx(10.0, abs=1e-9)      # 1 * acc of 10 minutes


# --- Input checks ------------------------------------------------------------
def test_bad_tables_raise():
    good = dict(n=[1], acc=[1.0], modes=[[R, 0, 0]], rides=[[10.0, 0, 0]],
                fts=[[0.0, 0.0]], egr=[1.0], freq=[1.0])

    def call(**change):
        a = {**good, **change}
        return C.path_costs(a["n"], a["acc"], a["modes"], a["rides"], a["fts"],
                            a["egr"], a["freq"], PAR)

    call()
    for bad in (dict(n=[0]), dict(n=[4]), dict(modes=[[C.CAR, 0, 0]]),
                dict(modes=[[-1, 0, 0]]), dict(rides=[[10.0, 0.0]]),
                dict(fts=[[0.0, 0.0, 0.0]]), dict(freq=[1.0, 2.0])):
        with pytest.raises(ValueError):
            call(**bad)


def test_empty_table():
    out = C.path_costs(np.zeros(0, int), np.zeros(0), np.zeros((0, 3), int),
                       np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0),
                       np.zeros(0), PAR)
    assert set(out) == set(KEYS)
    assert all(v.shape == (0,) for v in out.values())
