"""Identity and keep rules for the candidate paths of ONE FUA pair.

Plan step 4 and "Method in brief". The rules are the R1-R5 of
GTC/path-set-rules-2026-10-16.md (section 4) with the changes of the plan:

R3  margin, applied first: a candidate is cut only if it is outside the margin
    on BOTH the weighted time (gtt) and the weighted time plus hidden wait
    (gtc), each against the minimum of its mode class.
R1  identity: candidates of one mode sequence whose clock time T lies within
    the band of a leader are one alternative; the rest are its variants.
R2  keep: banded dominance on (transfers, time[, money]) within a mode class,
    with a frequency gate; the kept set is the stable set.
R4  no mode class loses its last alternative to the comparison across classes
    (guarantee) or to the cap.
R5  cap on the number of alternatives, by gtc.

Three quantities stay apart: T (clock minutes door to door), gtt (the deck's
weighted minutes, slide 229) and gtc (gtt plus hidden wait). Frequency never
enters T; it acts only through the gate and through gtc.

Adapted from the reference code GTC/evidence/path-set-rules-verify_example.py
(r1_merge, stable_set, r4_guarantee, r5_cap) and path-set-rules-verify_r2.py
(dom, kernel_elim). Differences from that code are stated in the docstrings.

``apply_rules`` takes one row per candidate with the columns ``INPUT``; ``M``
(money) is optional and may hold NaN. Inside, the candidates are column
arrays in increasing ``cand`` and a mode class is its position ("code") in
the sorted list of the classes present.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from numba import njit

from . import io

INPUT = ["cand", "seq", "n", "T", "gtt", "gtc", "freq", "first_link", "key",
         "cities"]
ALTERNATIVES = ["alt", "seq", "cls", "n", "T", "gtt", "gtc", "freq",
                "freq_pooled", "M", "n_variants", "members", "first_link",
                "key", "cities", "eps_T", "status"]
TRACE = ["cand", "seq", "cls", "alt", "status", "rule", "by", "detail"]

BANDS = ("EPS_T_ABS", "EPS_T_REL", "EPS_M")       # the only bands allowed
R1_KEYS = ("sequence", "class")
R2_TIMES = {"clock": "T", "weighted": "gtt"}      # compared time of R2
ORDERS = ("r3_first", "october")
KEPT = ("kept", "kept_guarantee")


# --- Configuration -----------------------------------------------------------
def check_params(params: dict) -> None:
    """Refuse a configuration under which the kept set is not unique.

    Only time and money may carry a band. The stable set of R2 exists and is
    unique because the transfer count is compared strictly and at most two
    attributes are banded; a band on the transfer count or on a third
    attribute (departure, arrival, transfer time ...) allows dominance cycles
    (GTC/reports/Path dominance criteria for 3MARS.md, "Two banded attributes
    are the ceiling"). Any parameter ``EPS_*`` other than ``BANDS`` with a
    non-zero value is therefore an error.

    RHO_F must lie in [0, 1]: above 1 the most frequent member of an R1 group
    would fail its own gate and the group would have no representative.
    """
    extra = sorted(k for k, v in params.items()
                   if k.startswith("EPS_") and k not in BANDS and v)
    if extra:
        raise ValueError(
            f"only time and money may carry a band; remove {extra}: a band on "
            "the transfer count or on a third attribute breaks the stable set")
    if params["R1_KEY"] not in R1_KEYS:
        raise ValueError(f"R1_KEY must be one of {R1_KEYS}")
    if params["R2_TIME"] not in R2_TIMES:
        raise ValueError(f"R2_TIME must be one of {tuple(R2_TIMES)}")
    if not 0 <= params["RHO_F"] <= 1:
        raise ValueError("RHO_F must lie in [0, 1]")
    if min(params["EPS_T_ABS"], params["EPS_T_REL"], params["EPS_M"] or 0,
           params["MU"]) < 0:
        raise ValueError("EPS_T_ABS, EPS_T_REL, EPS_M and MU must not be negative")


def mode_class(seq: str) -> str:
    """Mode class of a mode sequence: its set of modes, sorted ("RAB" -> "ABR").

    Plan, Definitions: "Mode class = set of modes used (car is its own class)".
    """
    return "".join(sorted(set(seq)))


def class_min(code: np.ndarray, value: np.ndarray, k: int) -> np.ndarray:
    """Minimum of ``value`` in each of the ``k`` mode classes."""
    out = np.full(k, np.inf)
    np.minimum.at(out, code, value)
    return out


def class_eps(code: np.ndarray, T: np.ndarray, k: int, params: dict) -> np.ndarray:
    """Time band of every mode class, in minutes.

    eps_T = max(EPS_T_ABS, EPS_T_REL x best T of the class), the best T taken
    over the candidates passed in (October proposal, parameter table). The
    same band serves R1 and R2 of that class.
    """
    return np.maximum(float(params["EPS_T_ABS"]),
                      params["EPS_T_REL"] * class_min(code, T, k))


def gate(freq, other, rho: float) -> np.ndarray:
    """Frequency gate: True where ``freq >= RHO_F x other``; RHO_F = 0 is off.

    Plan, thesis table: the relative form of the low-frequency correction of
    Cats (2011) eq. 5.5. Used for the representative of an R1 group and for
    the removals of R2.
    """
    if rho <= 0:
        return np.ones(np.broadcast(freq, other).shape, dtype=bool)
    return np.asarray(freq >= rho * other)


# --- R3 ----------------------------------------------------------------------
def margin(code: np.ndarray, value: np.ndarray, mu: float) -> np.ndarray:
    """(1 + MU) x the minimum of ``value`` in the row's mode class."""
    return (1 + mu) * class_min(code, value, int(code.max()) + 1)[code]


def r3_cut(code: np.ndarray, gtt: np.ndarray, gtc: np.ndarray, mu: float):
    """R3: a candidate is cut only if gtt > (1 + MU) x the class minimum of gtt
    AND gtc > (1 + MU) x the class minimum of gtc (plan step 4).

    A rare fast path (low gtt, high gtc) and a frequent slow one (the reverse)
    both pass. Returns (cut, bound on gtt, bound on gtc), row by row.
    """
    b_gtt, b_gtc = margin(code, gtt, mu), margin(code, gtc, mu)
    return (gtt > b_gtt) & (gtc > b_gtc), b_gtt, b_gtc


# --- R1 ----------------------------------------------------------------------
@njit(cache=True)
def _sweep(grp, T, M, eps_T, eps_M):
    """Position of the leader of every row; rows sorted by (group, T, gtt, key).

    Leaders of a group open in increasing T, so those still within the time
    band of the current row are a trailing window ``opened[lo:hi]``; a row
    joins the first of them. ``eps_M < 0`` switches the money condition off.
    """
    n = T.shape[0]
    leader = np.empty(n, dtype=np.int64)
    opened = np.empty(n, dtype=np.int64)
    lo = 0
    hi = 0
    for i in range(n):
        if i > 0 and grp[i] != grp[i - 1]:
            lo = hi
        while lo < hi and T[i] - T[opened[lo]] > eps_T[i]:
            lo += 1
        found = -1
        for j in range(lo, hi):
            k = opened[j]
            if (eps_M < 0 or np.isnan(M[i]) or np.isnan(M[k])
                    or abs(M[i] - M[k]) <= eps_M):
                found = k
                break
        if found < 0:
            opened[hi] = i
            hi += 1
            found = i
        leader[i] = found
    return leader


def r1_merge(c: dict, rows: np.ndarray, eps: np.ndarray, params: dict):
    """R1: group the candidates ``rows`` into alternatives (plan step 4).

    Within the R1 key, in increasing (T, gtt, key): a candidate joins the
    first leader whose T is within eps_T of its own, else becomes a leader.
    The key is the mode sequence; with R1_KEY = "class" it is the mode class
    together with the transfer count, so that "RB" and "BR" merge while a
    direct path never merges with a one-transfer path (the count carries no
    band). When the money band is on, a candidate joins a leader only if
    their money also lies within EPS_M where both are known (October R1).

    The representative of a group is its fastest member whose freq is at
    least RHO_F x the group's largest freq (ties: lower gtt, then key), so a
    faster variant is never hidden behind a slower representative unless it
    is rare.

    Returns (sweep, leader, alt): the rows in sweep order and, for every row
    of ``c``, the row of its leader and of its representative (-1 for rows
    not passed in).
    """
    if params["R1_KEY"] == "sequence":
        grp = c["seq_code"]
    else:
        grp = c["code"] * (int(c["n"].max()) + 1) + c["n"]
    sweep = rows[np.lexsort((c["cand"][rows], c["key_rank"][rows],
                             c["gtt"][rows], c["T"][rows], grp[rows]))]
    eps_M = params["EPS_M"]
    lead = _sweep(grp[sweep], c["T"][sweep], c["M"][sweep], eps[c["code"][sweep]],
                  -1.0 if eps_M is None else float(eps_M))
    freq = c["freq"][sweep]
    top = np.zeros(len(sweep))
    np.maximum.at(top, lead, freq)
    ok = np.flatnonzero(gate(freq, top[lead], params["RHO_F"]))
    rep = np.full(len(sweep), len(sweep), dtype=np.int64)
    np.minimum.at(rep, lead[ok], ok)
    leader = np.full(len(c["cand"]), -1, dtype=np.int64)
    alt = leader.copy()
    leader[sweep], alt[sweep] = sweep[lead], sweep[rep[lead]]
    return sweep, leader, alt


def pooled_freq(alt: np.ndarray, first_link: np.ndarray, freq: np.ndarray,
                size: int) -> np.ndarray:
    """Pooled frequency of every alternative, indexed by its row.

    Sum over the DISTINCT first links of the alternative of the largest freq
    among the members that start on that link. This stands in for the plan's
    "distinct first-leg departures over all variants": variants that share
    their first service link share its departures, so their frequencies are
    never added.
    """
    o = np.lexsort((first_link, alt))
    a, link = alt[o], first_link[o]
    start = np.flatnonzero(np.r_[True, (a[1:] != a[:-1]) | (link[1:] != link[:-1])])
    out = np.zeros(size)
    np.add.at(out, a[start], np.maximum.reduceat(freq[o], start))
    return out


# --- R2 ----------------------------------------------------------------------
def dominance(n, X, M, freq, pooled, eps_T: float, eps_M, rho: float) -> np.ndarray:
    """R2 relation: ``adj[p, q]`` is True when p removes q.

    p removes q iff n_p <= n_q, X_p <= X_q + eps_T, [M_p <= M_q + eps_M],
    p is noticeably better on at least one attribute (n_p < n_q, or
    X_p < X_q - eps_T, or M_p < M_q - eps_M) and freq_p >= RHO_F x
    pooled_q. The band sits on the dominated side (Delling et al. 2013 slack;
    16 Oct note, section 3.3 kind (ii)); the transfer count has no band.
    The money terms apply to a pair when ``eps_M`` is not None and both M are
    known. The caller restricts the arrays to one mode class (banded), or
    passes zero bands for the comparison across classes.
    """
    n, X, M = np.asarray(n), np.asarray(X, float), np.asarray(M, float)
    weak = (n[:, None] <= n[None, :]) & (X[:, None] <= X[None, :] + eps_T)
    notice = (n[:, None] < n[None, :]) | (X[:, None] < X[None, :] - eps_T)
    if eps_M is not None:
        both = ~np.isnan(M[:, None]) & ~np.isnan(M[None, :])
        weak &= ~both | (M[:, None] <= M[None, :] + eps_M)
        notice |= both & (M[:, None] < M[None, :] - eps_M)
    return weak & notice & gate(np.asarray(freq, float)[:, None],
                                np.asarray(pooled, float)[None, :], rho)


def stable_set(adj: np.ndarray):
    """Kept set of a removal relation by iterated elimination.

    Keep what nothing still in play removes, remove what the kept remove,
    repeat (16 Oct note, R2; ``stable_set`` and ``kernel_elim`` of the
    reference scripts). No kept element removes a kept one and every removed
    element is removed by a kept one; a kept element may be dominated by a
    removed one. Raises RuntimeError when the relation has a cycle, in which
    case no such set need exist.

    Returns (kept, by): a boolean mask and, for every removed element, the
    position of the first kept element that removes it (-1 for the kept).
    """
    alive = np.ones(adj.shape[0], dtype=bool)
    kept = np.zeros(adj.shape[0], dtype=bool)
    while alive.any():
        free = alive & ~adj[alive].any(axis=0)
        if not free.any():
            raise RuntimeError(
                f"dominance cycle among positions {np.flatnonzero(alive).tolist()}")
        kept |= free
        alive &= ~free & ~adj[free].any(axis=0)
    by = np.where(kept, -1, (adj & kept[:, None]).argmax(axis=0))
    return kept, by


def _relation(a: dict, idx: np.ndarray, x: str, eps_T: float, eps_M,
              rho: float) -> np.ndarray:
    """R2 relation among the alternatives ``idx``."""
    return dominance(a["n"][idx], a[x][idx], a["M"][idx], a["freq"][idx],
                     a["pooled"][idx], eps_T, eps_M, rho)


def _explain(a: dict, p: int, q: int, x: str, eps_T: float, eps_M,
             rho: float) -> str:
    """The numbers of the test "p removes q", for the trace."""
    parts = [f"n {a['n'][p]} <= {a['n'][q]}",
             f"{x} {a[x][p]:g} <= {a[x][q]:g} + {eps_T:g}"]
    if eps_M is not None and not np.isnan(a["M"][[p, q]]).any():
        parts.append(f"M {a['M'][p]:g} <= {a['M'][q]:g} + {eps_M:g}")
    if rho > 0:
        parts.append(f"freq {a['freq'][p]:g} >= {rho:g} x {a['pooled'][q]:g}")
    return "; ".join(parts)


# --- R4, R5 ------------------------------------------------------------------
def cap_drop(cls, cap: int) -> np.ndarray:
    """R5 subject to R4: which alternatives the cap drops.

    ``cls`` holds the mode classes of the alternatives in increasing gtc.
    While more than ``cap`` remain, the one with the highest gtc that is not
    the last of its mode class is dropped (``r5_cap`` of the reference
    script, with the protection always on).
    """
    left = Counter(cls)
    over = len(cls) - cap
    drop = np.zeros(len(cls), dtype=bool)
    for i in range(len(cls) - 1, -1, -1):
        if over <= 0:
            break
        if left[cls[i]] > 1:
            drop[i] = True
            left[cls[i]] -= 1
            over -= 1
    return drop


# --- The whole procedure -----------------------------------------------------
def _prepare(cands: pd.DataFrame):
    """Checked input as column arrays in increasing ``cand``.

    Adds ``M`` (NaN when the column is absent), ``key_rank`` (the order of
    the keys as numbers, for sorting), ``seq_code`` and ``code`` (position of
    the row's mode class in the sorted list of classes, which is returned
    beside the columns).
    """
    missing = [k for k in INPUT if k not in cands.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    order = np.argsort(cands["cand"].to_numpy(), kind="stable")
    c = {k: cands[k].to_numpy()[order] for k in ("seq", "cities")}
    for k in ("cand", "n", "first_link"):
        c[k] = cands[k].to_numpy(np.int64)[order]
    for k in ("T", "gtt", "gtc", "freq"):
        c[k] = cands[k].to_numpy(np.float64)[order]
    c["key"] = cands["key"].to_numpy()[order]
    c["key_rank"] = cands["key"].rank(method="dense").to_numpy()[order]
    c["M"] = (cands["M"].to_numpy(np.float64, na_value=np.nan)[order]
              if "M" in cands.columns else np.full(len(order), np.nan))
    if (c["cand"][1:] == c["cand"][:-1]).any():
        raise ValueError("candidate ids are not unique")
    if any(np.isnan(c[k]).any() for k in ("T", "gtt", "gtc", "freq")):
        raise ValueError("T, gtt, gtc and freq must be known for every candidate")
    c["seq_code"], seqs = pd.factorize(c["seq"], sort=True)
    of_seq = [mode_class(s) for s in seqs]
    classes = sorted(set(of_seq))
    c["code"] = np.array([classes.index(k) for k in of_seq],
                         dtype=np.int64)[c["seq_code"]]
    return c, classes


def apply_rules(cands: pd.DataFrame, params: dict | None = None,
                order: str = "r3_first", guarantee: bool = True):
    """Apply the rules to the candidates of one FUA pair.

    order = "r3_first" (plan): R3 on gtt and gtc, R1, R2, [comparison across
    classes], R4, R5. order = "october" (16 Oct note, for comparison runs):
    R1, R2, [across classes], then the margin on gtc alone among the kept
    alternatives of each class, R4, R5.

    Comparison across mode classes (CROSS_CLASS): the same test with all
    bands zero, on the survivors of R2, between alternatives of different
    classes. With ``guarantee`` on (R4), a class whose last alternative it
    removed gets back its survivor with the lowest gtc, as ``kept_guarantee``.

    Money: within a class the money band is used only when M is known for
    every alternative of the class; with M known for some only, the banded
    relation can cycle. Across classes (zero bands) it is used pair by pair.

    Cap (R5): max(K_CAP, mode classes among the candidates + K_EXTRA).

    Returns (alternatives, trace).
    alternatives: the kept set, columns ``ALTERNATIVES``, in increasing
        (gtc, key). ``alt`` is the candidate id of the representative, whose
        n, T, gtt, gtc, freq and M the alternative carries; ``members`` are
        the candidate ids in increasing (T, gtt, key), ``n_variants`` the
        members other than the representative; ``status`` is kept |
        kept_guarantee.
    trace: one row per candidate in increasing ``cand``, columns ``TRACE``;
        ``status`` is kept | kept_guarantee | variant_of:<alt> |
        removed_by:<alt> | cut_r3 | capped; ``alt`` the alternative the
        candidate belongs to (-1 when cut before R1); ``by`` the candidate
        on the other side of the deciding test (leader, remover; -1 if none)
        and ``detail`` the numbers of that test. Variants keep
        ``variant_of`` whatever happens to their alternative.
    """
    params = io.P if params is None else params
    check_params(params)
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    c, classes = _prepare(cands)
    size = len(c["cand"])
    if size == 0:
        return pd.DataFrame(columns=ALTERNATIVES), pd.DataFrame(columns=TRACE)
    mu, rho, eps_M = params["MU"], params["RHO_F"], params["EPS_M"]
    x = R2_TIMES[params["R2_TIME"]]
    cand, code, T, gtt, gtc, freq = (
        c[k] for k in ("cand", "code", "T", "gtt", "gtc", "freq"))
    status = np.full(size, "kept", dtype=object)
    rule = np.full(size, "", dtype=object)
    detail = np.full(size, "", dtype=object)
    by = np.full(size, -1, dtype=np.int64)

    # R3 on the candidates
    live = np.ones(size, dtype=bool)
    if order == "r3_first":
        cut, b_gtt, b_gtc = r3_cut(code, gtt, gtc, mu)
        status[cut], rule[cut] = "cut_r3", "R3"
        detail[cut] = [
            f"gtt {g:g} > {b:g}; gtc {h:g} > {d:g} ({1 + mu:g} x class minimum)"
            for g, b, h, d in zip(gtt[cut].tolist(), b_gtt[cut].tolist(),
                                  gtc[cut].tolist(), b_gtc[cut].tolist())]
        live = ~cut

    # R1: alternatives and their variants
    rows = np.flatnonzero(live)
    eps = class_eps(code[rows], T[rows], len(classes), params)
    sweep, leader, alt = r1_merge(c, rows, eps, params)
    var = rows[alt[rows] != rows]
    status[var] = np.char.add("variant_of:", cand[alt[var]].astype(str))
    rule[var], by[var] = "R1", cand[leader[var]]
    detail[var] = [f"T {t:g} - {u:g} <= {e:g}" for t, u, e in zip(
        T[var].tolist(), T[leader[var]].tolist(), eps[code[var]].tolist())]
    # ... and why a member as fast as the representative does not represent
    top = np.zeros(size)
    np.maximum.at(top, alt[rows], freq[rows])
    rare = var[~gate(freq[var], top[alt[var]], rho) & (T[var] <= T[alt[var]])]
    detail[rare] = [f"{d}; freq {f:g} < {rho:g} x {t:g}" for d, f, t in zip(
        detail[rare], freq[rare].tolist(), top[alt[rare]].tolist())]

    # The alternatives, in increasing (gtc, key): positions 0 .. len(reps) - 1
    reps = np.unique(alt[rows])
    reps = reps[np.lexsort((cand[reps], c["key_rank"][reps], gtc[reps]))]
    a = {k: c[k][reps] for k in ("n", "T", "gtt", "gtc", "M", "freq", "code")}
    a["pooled"] = pooled_freq(alt[rows], c["first_link"][rows], freq[rows],
                              size)[reps]
    state = np.full(len(reps), "kept", dtype=object)
    why = np.full(len(reps), "", dtype=object)
    numbers = np.full(len(reps), "", dtype=object)
    other = np.full(len(reps), -1, dtype=np.int64)

    # R2 within each mode class
    for k in np.unique(a["code"]):
        idx = np.flatnonzero(a["code"] == k)
        band_M = None if np.isnan(a["M"][idx]).any() else eps_M
        kept, first = stable_set(_relation(a, idx, x, eps[k], band_M, rho))
        for q, p in zip(idx[~kept], idx[first[~kept]]):
            state[q], why[q], other[q] = "removed_by", "R2", p
            numbers[q] = _explain(a, p, q, x, eps[k], band_M, rho)

    # Strict comparison across mode classes, subject to R4
    if params["CROSS_CLASS"]:
        idx = np.flatnonzero(state == "kept")
        zero_M = None if eps_M is None else 0
        mine = a["code"][idx]
        kept, first = stable_set(_relation(a, idx, x, 0, zero_M, rho)
                                 & (mine[:, None] != mine[None, :]))
        for q, p in zip(idx[~kept], idx[first[~kept]]):
            state[q], why[q], other[q] = "removed_by", "R2-cross", p
            numbers[q] = _explain(a, p, q, x, 0, zero_M, rho)
        if guarantee:
            for k in np.unique(mine[~kept]):
                if not kept[mine == k].any():
                    best = idx[mine == k][0]
                    state[best], why[best] = "kept_guarantee", "R4"
                    numbers[best] = f"last of class {classes[k]}; {numbers[best]}"

    # October order: the margin on gtc alone, among the kept alternatives
    if order == "october":
        idx = np.flatnonzero(np.isin(state, KEPT))
        bound = margin(a["code"][idx], a["gtc"][idx], mu)
        for q, b in zip(idx, bound):
            if a["gtc"][q] > b:
                state[q], why[q], other[q] = "cut_r3", "R3", -1
                numbers[q] = (f"gtc {a['gtc'][q]:g} > {b:g} ({1 + mu:g} x class "
                              "minimum of the kept)")

    # R5 subject to R4
    idx = np.flatnonzero(np.isin(state, KEPT))
    cap = max(params["K_CAP"], len(classes) + params["K_EXTRA"])
    for rank in np.flatnonzero(cap_drop(a["code"][idx].tolist(), cap)):
        q = idx[rank]
        state[q], why[q] = "capped", "R5"
        numbers[q] = f"gtc {a['gtc'][q]:g}; rank {rank + 1} of {len(idx)}; cap {cap}"

    # Status of the representatives; the kept set
    status[reps] = [f"{s}:{cand[reps[p]]}" if s == "removed_by" else s
                    for s, p in zip(state, other)]
    rule[reps], detail[reps] = why, numbers
    by[reps] = np.where(other >= 0, cand[reps[other]], -1)
    final = np.flatnonzero(np.isin(state, KEPT))
    r = reps[final]
    members = [cand[sweep[alt[sweep] == q]].tolist() for q in r]
    names = np.array(classes, dtype=object)
    alts = pd.DataFrame({
        "alt": cand[r], "seq": c["seq"][r], "cls": names[code[r]],
        "n": c["n"][r], "T": T[r], "gtt": gtt[r], "gtc": gtc[r],
        "freq": freq[r], "freq_pooled": a["pooled"][final], "M": c["M"][r],
        "n_variants": [len(m) - 1 for m in members], "members": members,
        "first_link": c["first_link"][r], "key": c["key"][r],
        "cities": c["cities"][r], "eps_T": eps[code[r]], "status": state[final]})
    trace = pd.DataFrame({
        "cand": cand, "seq": c["seq"], "cls": names[code],
        "alt": np.where(alt >= 0, cand[alt], -1), "status": status,
        "rule": rule, "by": by, "detail": detail})
    return alts, trace
