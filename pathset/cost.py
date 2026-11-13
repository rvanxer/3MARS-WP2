"""Generalised travel time of a path (deck slides 228-229; plan "Three
quantities, kept apart").

A path p = [e0, e1 ... en, en+1] has a car access of ``acc`` minutes (e0),
n service links with riding times ``ride_i`` and modes ``m_i``, n - 1
transfers of ``ft_i`` minutes from arrival to departure (any move between hubs
included) and a car egress of ``egr`` minutes (en+1). Three quantities:

    T    clock time door to door, the deck's terms without weights
         T   = acc + TAU_DEP[m_1] + sum(ride_i) + sum(ft_i) + TAU_ARR[m_n] + egr
    gtt  the deck's weighted sum (slide 229)
         gtt = t_a + t_dep + t_ivt + t_trans + t_tp + t_arr + t_e
             t_a     = BETA_ACCESS * acc
             t_dep   = TAU_DEP[m_1]
             t_ivt   = sum(BETA_IVT[m_i] * ride_i)
             t_trans = BETA_TRANS * sum(ft_i)
             t_tp    = TAU_TP * (n - 1)
             t_arr   = TAU_ARR[m_n]
             t_e     = BETA_EGRESS * egr
    gtc  gtt + hidden wait, the time bracket of the deck's GTC (slide 228)
         hwt = min(BETA_HWT / (2 * freq), HWT_MAX), freq in departures per day;
               freq <= 0 or NaN gives HWT_MAX

Car, the direct alternative: T = minutes, gtt = BETA_IVT["C"] * minutes, no
allowances, hwt = 0.

Lower bounds for the search (plan step 3): the same formulas with the minimum
transfer times in place of ft_i, and hwt at the largest daily count (freq_max)
of the chain's sparsest link. Every term grows with the transfer times, so the
bounds of T and gtt never exceed the true value. The bound of hwt, and with it
that of gtc, holds only while the path frequency is not above that count, and
the counting of opportunities (``pathset.transfer``) does not guarantee this:
a typical day can hold departures that never run on the same date, and a day
of first departure can use two calendar runs of a link that leaves after
midnight. Rule: the frequency of a chain enters the hidden wait through
``capped_frequency`` (the counted frequency, at most the sparsest link's
freq_max); with it the bound holds whatever the counting basis. The search
checks every candidate it keeps with ``bounds_hold`` and stops on a failure.

Two forms with identical formulas and the same order of operations:
vectorised numpy functions for tables of paths with up to ``MAX_LINKS`` links,
and scalar numba kernels (``k_*``) for use inside the search. Both read every
parameter from the array made by ``param_array``; nothing is read from
``pathset.io.P`` at import time.

Compiled callers. numba keys a disk cache on the file that defines the cached
function, not on the files of the functions it calls. A function compiled with
``cache=True`` in another file that calls a kernel keeps the old formula after
this file is edited, without a warning (reproduced with numba 0.65). Rule: no
compiled function outside this file that reaches a ``k_*`` kernel, directly or
through other compiled functions, uses ``cache=True``; it is compiled once per
process instead. tests/test_cost.py scans the repository for offenders. The
kernels themselves are cached: their cache is keyed on this file, which is why
the mode codes below are written out and not derived from another file.
Parameter changes are safe in any case, because ``par`` is an argument.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .schemas import MODES

# Mode codes: A = 0, B = 1, R = 2 as in the data contract (schemas.MODES; the
# agreement is tested), car = 3. Compiled into the kernels through the layout
MODE_CODE = {"A": 0, "B": 1, "R": 2, "C": 3}
N_MODES = len(MODE_CODE)

CAR = MODE_CODE["C"]

# Link slots of the table form and of the kernels
MAX_LINKS = 3

# Layout of the parameter array: three blocks indexed by mode code, six scalars
IVT0 = 0                      # par[IVT0 + m] = BETA_IVT of mode m
DEP0 = IVT0 + N_MODES         # par[DEP0 + m] = TAU_DEP of mode m (car 0)
ARR0 = DEP0 + N_MODES         # par[ARR0 + m] = TAU_ARR of mode m (car 0)
I_ACCESS = ARR0 + N_MODES
I_EGRESS = I_ACCESS + 1
I_TRANS = I_ACCESS + 2
I_TP = I_ACCESS + 3
I_HWT = I_ACCESS + 4
I_HWT_MAX = I_ACCESS + 5
N_PAR = I_ACCESS + 6

# Weighted components of gtt, in the deck's order (slide 229)
COMPONENTS = ("t_a", "t_dep", "t_ivt", "t_trans", "t_tp", "t_arr", "t_e")

# Quantities with a lower bound, in the order returned by the kernels
BOUNDED = ("T", "gtt", "hwt", "gtc")


def param_array(P: dict) -> np.ndarray:
    """Parameters of the generalised travel time as one float64 array.

    Rule: values of ``BETA_*``, ``TAU_*`` and ``HWT_MAX`` of the parameter
    dict (``pathset.io.P`` or a copy), placed by the layout constants of this
    module. The car has a riding-time weight and no allowances (its TAU_DEP
    and TAU_ARR slots are 0). Source: deck slide 229; plan "Parameters".
    """
    par = np.zeros(N_PAR, dtype=np.float64)
    for mode, code in MODE_CODE.items():
        par[IVT0 + code] = P["BETA_IVT"][mode]
    for mode in MODES:
        par[DEP0 + MODE_CODE[mode]] = P["TAU_DEP"][mode]
        par[ARR0 + MODE_CODE[mode]] = P["TAU_ARR"][mode]
    par[I_ACCESS] = P["BETA_ACCESS"]
    par[I_EGRESS] = P["BETA_EGRESS"]
    par[I_TRANS] = P["BETA_TRANS"]
    par[I_TP] = P["TAU_TP"]
    par[I_HWT] = P["BETA_HWT"]
    par[I_HWT_MAX] = P["HWT_MAX"]
    return par


# --- Tables of paths (numpy) -------------------------------------------------
def hidden_wait(freq, par: np.ndarray) -> np.ndarray:
    """Hidden waiting time, minutes, for frequencies in departures per day.

    Rule: min(BETA_HWT / (2 * freq), HWT_MAX); freq <= 0 or NaN gives HWT_MAX.
    Source: deck slide 228 (form), plan "Parameters" (value and cap).
    """
    freq = np.asarray(freq, dtype=np.float64)
    served = freq > 0
    wait = par[I_HWT] / (2.0 * np.where(served, freq, 1.0))
    return np.where(served, np.minimum(wait, par[I_HWT_MAX]), par[I_HWT_MAX])


def _used_slots(n: np.ndarray) -> np.ndarray:
    """Boolean (N, MAX_LINKS): link slot j of row i is in use when j < n[i]."""
    if n.ndim != 1:
        raise ValueError(f"link counts must be one-dimensional, got {n.shape}")
    if n.size and (n.min() < 1 or n.max() > MAX_LINKS):
        raise ValueError(f"link counts must lie in 1..{MAX_LINKS}")
    return np.arange(MAX_LINKS) < n[:, None]


def _check_shape(name: str, arr: np.ndarray, shape: tuple) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")


def path_costs(n, acc, modes, rides, fts, egr, freq, par: np.ndarray) -> dict:
    """T, gtt, hwt, gtc and the seven components of a table of paths.

    n       (N,)   number of service links, 1..MAX_LINKS
    acc     (N,)   access minutes (e0)
    modes   (N, 3) mode codes of the links
    rides   (N, 3) riding minutes of the links
    fts     (N, 2) transfer minutes, arrival to departure; fts[:, j] lies
                   between link j and link j + 1
    egr     (N,)   egress minutes (en+1)
    freq    (N,)   path frequency, departures per day
    Slots beyond the link count are ignored, whatever they hold.

    Rule: the formulas of the module docstring (deck slides 228-229). Returns
    a dict of float64 arrays keyed ``COMPONENTS`` + T, gtt, hwt, gtc.
    """
    n = np.asarray(n, dtype=np.int64)
    acc = np.asarray(acc, dtype=np.float64)
    egr = np.asarray(egr, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.int64)
    rides = np.asarray(rides, dtype=np.float64)
    fts = np.asarray(fts, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    used = _used_slots(n)
    _check_shape("modes", modes, (n.size, MAX_LINKS))
    _check_shape("rides", rides, (n.size, MAX_LINKS))
    _check_shape("fts", fts, (n.size, MAX_LINKS - 1))
    for name, arr in (("acc", acc), ("egr", egr), ("freq", freq)):
        _check_shape(name, arr, n.shape)
    if ((modes < 0) | (modes >= len(MODES)))[used].any():
        raise ValueError("a service link must have mode code 0 (A), 1 (B) or 2 (R)")

    m = np.where(used, modes, 0)
    ride = np.where(used, rides, 0.0)
    ft = np.where(used[:, 1:], fts, 0.0)
    weighted = par[IVT0 + m] * ride
    ride_sum = ride[:, 0] + ride[:, 1] + ride[:, 2]
    ft_sum = ft[:, 0] + ft[:, 1]
    m_last = m[np.arange(n.size), n - 1]

    out = {
        "t_a": par[I_ACCESS] * acc,
        "t_dep": par[DEP0 + m[:, 0]],
        "t_ivt": weighted[:, 0] + weighted[:, 1] + weighted[:, 2],
        "t_trans": par[I_TRANS] * ft_sum,
        "t_tp": par[I_TP] * (n - 1),
        "t_arr": par[ARR0 + m_last],
        "t_e": par[I_EGRESS] * egr,
    }
    out["T"] = acc + out["t_dep"] + ride_sum + ft_sum + out["t_arr"] + egr
    gtt = out["t_a"]
    for name in COMPONENTS[1:]:
        gtt = gtt + out[name]
    out["gtt"] = gtt
    out["hwt"] = hidden_wait(freq, par)
    out["gtc"] = gtt + out["hwt"]
    return out


def sparsest(n, counts) -> np.ndarray:
    """Smallest of the daily counts over the links in use, per row.

    Rule: with ``counts`` = freq_max of the links (N, 3) this is the largest
    number of runs of the chain's sparsest link on one date (plan step 3,
    "Bounds"). It is NOT by itself an upper bound of the counted path
    frequency: opportunities counted on typical days, or on a day of first
    departure that reaches two calendar runs of a link after midnight, can
    exceed it. ``capped_frequency`` makes it one.
    """
    n = np.asarray(n, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.float64)
    used = _used_slots(n)
    _check_shape("counts", counts, (n.size, MAX_LINKS))
    return np.where(used, counts, np.inf).min(axis=1)


def capped_frequency(n, freq, counts) -> np.ndarray:
    """Frequency of a chain as it enters the hidden wait, per row.

    Rule: min(counted frequency ``freq`` (N,), ``sparsest(n, counts)``) with
    ``counts`` = freq_max of the links (N, 3); NaN stays NaN. A chain runs no
    more often than its sparsest link (plan "Definitions": four feeders into
    one daily trunk give one a day), and a count above that comes from the
    counting basis, not from the timetable (module docstring). With this
    frequency in ``path_costs`` the hidden-wait bound of ``path_bounds`` holds
    by construction.
    """
    freq = np.asarray(freq, dtype=np.float64)
    cap = sparsest(n, counts)
    _check_shape("freq", freq, cap.shape)
    return np.minimum(freq, cap)


def path_bounds(n, acc, modes, rides, ft_min, egr, freq_max, par: np.ndarray) -> dict:
    """Lower bounds of T, gtt, hwt and gtc of a table of chains.

    Rule: ``path_costs`` with the minimum transfer times ``ft_min`` (N, 2) and
    the hidden wait at the largest daily count ``freq_max`` (N, 3) of the
    chain's sparsest link (plan step 3). The bounds of T and gtt hold for any
    transfer times not below ``ft_min``. Those of hwt and gtc hold only for a
    path frequency not above ``sparsest(n, freq_max)``: compare them with true
    values computed from ``capped_frequency``, never from a raw count.
    """
    low = path_costs(n, acc, modes, rides, ft_min, egr, sparsest(n, freq_max), par)
    return {name: low[name] for name in BOUNDED}


def bounds_hold(low: dict, true: dict) -> np.ndarray:
    """True per row when no lower bound exceeds the true value.

    Rule: low <= true for T, gtt, hwt and gtc, with ``low`` from
    ``path_bounds`` and ``true`` from ``path_costs``. No tolerance: both sides
    come from the same formulas and floating-point sums and products are
    monotone. A NaN on either side fails. Source: plan step 3, the bounds
    "never exceed the true value"; the search tests that claim on every
    candidate it keeps.
    """
    ok = np.ones(np.shape(low[BOUNDED[0]]), dtype=bool)
    for name in BOUNDED:
        ok &= np.asarray(low[name]) <= np.asarray(true[name])
    return ok


def car_costs(minutes, par: np.ndarray) -> dict:
    """The direct car alternative, with the keys of ``path_costs``.

    Rule: T = minutes; gtt = BETA_IVT["C"] * minutes, all of it riding time
    (t_ivt); no allowances, no access or egress, hwt = 0 (plan "Parameters":
    car as a direct alternative, not in the deck).
    """
    minutes = np.asarray(minutes, dtype=np.float64)
    zero = np.zeros_like(minutes)
    out = {name: zero.copy() for name in COMPONENTS}
    out["t_ivt"] = par[IVT0 + CAR] * minutes
    out["T"] = minutes.copy()
    out["gtt"] = out["t_ivt"].copy()
    out["hwt"] = zero.copy()
    out["gtc"] = out["gtt"].copy()
    return out


# --- Scalar kernels (numba) --------------------------------------------------
@njit(cache=True)
def k_hidden_wait(freq, par):
    """Scalar ``hidden_wait``: min(BETA_HWT / (2 * freq), HWT_MAX); freq <= 0
    or NaN gives HWT_MAX (deck slide 228)."""
    if not freq > 0:
        return par[I_HWT_MAX]
    return min(par[I_HWT] / (2.0 * freq), par[I_HWT_MAX])


@njit(cache=True)
def _k_sums(n, m1, m2, m3, r1, r2, r3, ft1, ft2, par):
    """Last mode, riding time, weighted riding time and transfer time of the
    n links in use; slots beyond n are not read."""
    m_last = m1
    ride_sum = 1.0 * r1
    ride_w = par[IVT0 + m1] * r1
    ft_sum = 0.0
    if n >= 2:
        m_last = m2
        ride_sum += r2
        ride_w += par[IVT0 + m2] * r2
        ft_sum += ft1
    if n >= 3:
        m_last = m3
        ride_sum += r3
        ride_w += par[IVT0 + m3] * r3
        ft_sum += ft2
    return m_last, ride_sum, ride_w, ft_sum


@njit(cache=True)
def k_components(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par):
    """(t_a, t_dep, t_ivt, t_trans, t_tp, t_arr, t_e) of one path with n
    links, weighted as on deck slide 229. ft1 lies between links 1 and 2, ft2
    between links 2 and 3; n must be 1..3 and the modes A, B or R (not
    checked)."""
    m_last, _, ride_w, ft_sum = _k_sums(n, m1, m2, m3, r1, r2, r3, ft1, ft2, par)
    return (par[I_ACCESS] * acc, par[DEP0 + m1], ride_w, par[I_TRANS] * ft_sum,
            par[I_TP] * (n - 1), par[ARR0 + m_last], par[I_EGRESS] * egr)


@njit(cache=True)
def k_gtt(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par):
    """Generalised travel time of one path: the sum of ``k_components`` in the
    deck's order (slide 229). With minimum transfer times it is the lower
    bound of the search."""
    t_a, t_dep, t_ivt, t_trans, t_tp, t_arr, t_e = k_components(
        n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par)
    return t_a + t_dep + t_ivt + t_trans + t_tp + t_arr + t_e


@njit(cache=True)
def k_clock(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par):
    """Clock time T of one path: access, departure allowance, riding,
    transfers, arrival allowance, egress, without weights and without the
    transfer penalty (plan "Three quantities")."""
    m_last, ride_sum, _, ft_sum = _k_sums(n, m1, m2, m3, r1, r2, r3, ft1, ft2, par)
    return acc + par[DEP0 + m1] + ride_sum + ft_sum + par[ARR0 + m_last] + egr


@njit(cache=True)
def k_costs(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, freq, par):
    """(T, gtt, hwt, gtc) of one path with frequency ``freq`` per day."""
    gtt = k_gtt(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par)
    hwt = k_hidden_wait(freq, par)
    return (k_clock(n, acc, m1, m2, m3, r1, r2, r3, ft1, ft2, egr, par),
            gtt, hwt, gtt + hwt)


@njit(cache=True)
def k_sparsest(n, c1, c2, c3):
    """Smallest of the daily counts of the n links in use (scalar
    ``sparsest``). With freq_max it bounds the path frequency only after
    ``k_capped_frequency``."""
    c = 1.0 * c1
    if n >= 2 and c2 < c:
        c = 1.0 * c2
    if n >= 3 and c3 < c:
        c = 1.0 * c3
    return c


@njit(cache=True)
def k_capped_frequency(n, freq, c1, c2, c3):
    """Frequency of a chain as it enters the hidden wait (scalar
    ``capped_frequency``): the counted frequency, at most the largest daily
    count (c1..c3 = freq_max) of the sparsest link; NaN stays NaN."""
    cap = k_sparsest(n, c1, c2, c3)
    if freq > cap:
        return cap
    return 1.0 * freq


@njit(cache=True)
def k_bounds(n, acc, m1, m2, m3, r1, r2, r3, ftmin1, ftmin2, egr, c1, c2, c3, par):
    """Lower bounds (T, gtt, hwt, gtc) of one chain: ``k_costs`` with the
    minimum transfer times and the hidden wait at the largest daily count
    (c1..c3 = freq_max) of the sparsest link (plan step 3). T and gtt are
    bounds for any transfer times not below the minimum; hwt and gtc only for
    a path frequency not above that count, so the true ``k_costs`` must take
    its frequency from ``k_capped_frequency``."""
    return k_costs(n, acc, m1, m2, m3, r1, r2, r3, ftmin1, ftmin2, egr,
                   k_sparsest(n, c1, c2, c3), par)


@njit(cache=True)
def k_bounds_hold(low, true):
    """Scalar ``bounds_hold``: True when no element of ``low`` (the tuple of
    ``k_bounds``) exceeds its counterpart in ``true`` (the tuple of
    ``k_costs``); a NaN fails. For the search to call on every candidate it
    keeps, stopping on False."""
    return (low[0] <= true[0] and low[1] <= true[1]
            and low[2] <= true[2] and low[3] <= true[3])


@njit(cache=True)
def k_car_gtt(minutes, par):
    """Generalised travel time of the direct car alternative:
    BETA_IVT["C"] * minutes. Its T is ``minutes`` and its hidden wait 0."""
    return par[IVT0 + CAR] * minutes
