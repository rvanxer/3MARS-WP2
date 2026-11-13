"""Review cases for pathset.cost: an independent form of the deck's formulas
(slides 228-229) written from the parameter dict, the kernels called from
another compiled function with the dtypes of the data contract, and the
condition under which the hidden-wait bound holds."""

import sys
from pathlib import Path

import numpy as np
import pytest
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import cost as C  # noqa: E402
from pathset import io  # noqa: E402

LETTER = {0: "A", 1: "B", 2: "R"}
OUT = ("T", "gtt", "hwt", "gtc")


def by_the_deck(n, acc, modes, rides, fts, egr, freq, P):
    """Slides 228-229 read straight from the parameter dict. Shares neither
    the parameter array nor its layout with the module."""
    seq = [LETTER[int(m)] for m in modes[:n]]
    ride = [float(r) for r in rides[:n]]
    ft = [float(f) for f in fts[:n - 1]]
    gtt = (P["BETA_ACCESS"] * acc + P["TAU_DEP"][seq[0]]
           + sum(P["BETA_IVT"][m] * r for m, r in zip(seq, ride))
           + P["BETA_TRANS"] * sum(ft) + P["TAU_TP"] * (n - 1)
           + P["TAU_ARR"][seq[-1]] + P["BETA_EGRESS"] * egr)
    T = acc + P["TAU_DEP"][seq[0]] + sum(ride) + sum(ft) + P["TAU_ARR"][seq[-1]] + egr
    hwt = min(P["BETA_HWT"] / (2 * freq), P["HWT_MAX"]) if freq > 0 else P["HWT_MAX"]
    return T, gtt, hwt, gtt + hwt


def table(seed, size):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 4, size)
    modes = rng.integers(0, 3, (size, 3))
    rides = rng.uniform(1, 1440, (size, 3))
    fts = rng.uniform(0, 300, (size, 2))
    acc = rng.uniform(0, 120, size)
    egr = rng.uniform(0, 120, size)
    freq = rng.uniform(0.01, 80, size)
    freq[::13] = 0.0
    freq[5::17] = np.nan
    return n, acc, modes, rides, fts, egr, freq


@njit
def _kernel_rows(n, acc, modes, rides, fts, egr, freq, par):
    """k_costs for every row, called the way the search will call it."""
    out = np.empty((n.size, 4))
    for i in range(n.size):
        out[i] = C.k_costs(n[i], acc[i], modes[i, 0], modes[i, 1], modes[i, 2],
                           rides[i, 0], rides[i, 1], rides[i, 2],
                           fts[i, 0], fts[i, 1], egr[i], freq[i], par)
    return out


@njit
def _bound_rows(n, acc, modes, rides, fts, egr, counts, par):
    out = np.empty((n.size, 4))
    for i in range(n.size):
        out[i] = C.k_bounds(n[i], acc[i], modes[i, 0], modes[i, 1], modes[i, 2],
                            rides[i, 0], rides[i, 1], rides[i, 2],
                            fts[i, 0], fts[i, 1], egr[i],
                            counts[i, 0], counts[i, 1], counts[i, 2], par)
    return out


def test_both_forms_equal_the_deck_formulas_written_from_the_dict():
    P = io.P
    par = C.param_array(P)
    n, acc, modes, rides, fts, egr, freq = table(77, 4000)
    tab = C.path_costs(n, acc, modes, rides, fts, egr, freq, par)
    for i in range(n.size):
        want = by_the_deck(int(n[i]), acc[i], modes[i], rides[i], fts[i], egr[i],
                           freq[i], P)
        got = C.k_costs(int(n[i]), acc[i], *modes[i], *rides[i], *fts[i], egr[i],
                        freq[i], par)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        np.testing.assert_allclose([tab[k][i] for k in OUT], want, rtol=1e-12)


def test_every_mode_sequence_of_one_to_three_links():
    P = io.P
    par = C.param_array(P)
    for n in (1, 2, 3):
        for m in np.ndindex(3, 3, 3):
            want = by_the_deck(n, 7.0, m, (11.0, 13.0, 17.0), (19.0, 23.0), 29.0, 3.0, P)
            got = C.k_costs(n, 7.0, *m, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 3.0, par)
            np.testing.assert_allclose(got, want, rtol=1e-12, err_msg=f"{n} {m}")


def test_kernels_inside_a_compiled_loop_with_contract_dtypes():
    # ps-links: ride float32, freq float32, freq_max int16; ps-events: ride
    # int16; ps-transfers: ft_mean, ft_min float32; mode codes int8
    par = C.param_array(io.P)
    n, acc, modes, rides, fts, egr, freq = table(3, 20000)
    counts = np.random.default_rng(4).integers(1, 300, (n.size, 3)).astype(np.int16)
    for ride_type in (np.float32, np.int16):
        a = (n.astype(np.int8), acc.astype(np.float32), modes.astype(np.int8),
             rides.astype(ride_type), fts.astype(np.float32), egr.astype(np.float32))
        f = freq.astype(np.float32)
        want = C.path_costs(*a, f, par)
        np.testing.assert_allclose(_kernel_rows(*a, f, par),
                                   np.stack([want[k] for k in OUT], axis=1), rtol=1e-12)
        low = C.path_bounds(*a, counts, par)
        np.testing.assert_allclose(_bound_rows(*a, counts, par),
                                   np.stack([low[k] for k in OUT], axis=1), rtol=1e-12)


def test_clock_time_has_no_weight_and_no_transfer_penalty():
    # T is the sum of the raw minutes plus the two allowances, whatever the
    # weights and the penalty are (plan "Three quantities")
    P = {**io.P, "BETA_ACCESS": 9.0, "BETA_EGRESS": 9.0, "BETA_TRANS": 9.0,
         "TAU_TP": 99, "BETA_IVT": {"A": 9.0, "B": 9.0, "R": 9.0, "C": 9.0}}
    par = C.param_array(P)
    T = C.k_clock(3, 10, 2, 0, 1, 25, 60, 120, 12, 20, 15, par)
    assert T == 10 + P["TAU_DEP"]["R"] + 205 + 32 + P["TAU_ARR"]["B"] + 15
    assert C.car_costs([300.0], par)["T"][0] == 300.0


def test_hidden_wait_bound_holds_only_up_to_the_sparsest_count():
    """The bound is BETA_HWT / (2 * min freq_max). It is a lower bound exactly
    when the path frequency does not exceed the sparsest link's largest daily
    count; whoever counts opportunities (typical days, chains running past
    midnight) has to guarantee that or clamp the frequency."""
    par = C.param_array(io.P)
    cap, beta = io.P["HWT_MAX"], io.P["BETA_HWT"]
    chain = (2, 10.0, 2, 2, 0, 60.0, 90.0, 0.0, 10.0, 0.0, 15.0)
    low = C.k_bounds(*chain, 20, 1, 0, par)[2]          # sparsest link: 1 a day
    assert low == min(beta / 2.0, cap)
    assert C.k_costs(*chain, 1.0, par)[2] >= low        # frequency 1: holds
    assert C.k_costs(*chain, 0.5, par)[2] >= low
    if beta / 4.0 < cap:                                # frequency 2 > 1: broken
        assert C.k_costs(*chain, 2.0, par)[2] < low
    # the fix: the counted frequency enters the hidden wait capped
    capped = C.k_capped_frequency(2, 2.0, 20, 1, 0)
    assert capped == 1.0
    assert C.k_costs(*chain, capped, par)[2] >= low


def test_table_form_rejects_car_inside_a_path_and_kernel_takes_plain_ints():
    par = C.param_array(io.P)
    with pytest.raises(ValueError):
        C.path_costs([2], [1.0], [[2, C.CAR, 0]], [[10.0, 10.0, 0.0]], [[5.0, 0.0]],
                     [1.0], [1.0], par)
    a = C.k_costs(2, 10, 2, 1, 0, 60, 90, 0, 10, 0, 15, 3, par)
    b = C.k_costs(2, 10.0, 2, 1, 0, 60.0, 90.0, 0.0, 10.0, 0.0, 15.0, 3.0, par)
    assert a == b
