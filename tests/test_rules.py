"""Identity and keep rules (pathset/rules.py): fixtures of the 16 Oct note and
of the plan's verification list, and properties on random instances."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import io, rules  # noqa: E402

# The rule parameters are fixed here so that the tests do not move with the
# parameter file
BASE = {**io.P, "EPS_T_ABS": 15, "EPS_T_REL": 0.05, "EPS_M": None, "MU": 0.5,
        "K_CAP": 8, "K_EXTRA": 3, "RHO_F": 0.5, "CROSS_CLASS": False,
        "R1_KEY": "sequence", "R2_TIME": "clock"}


def make(rows):
    """Candidates from short dicts: cand, seq and T are required. Defaults:
    n = len(seq) - 1, gtt = T, gtc = gtt, freq = 1 (car: inf), a first link
    of its own (car: -1), key = the id, no money."""
    out = []
    for r in rows:
        car = r["seq"] == "C"
        gtt = r.get("gtt", r["T"])
        out.append({
            "cand": r["cand"], "seq": r["seq"],
            "n": r.get("n", 0 if car else len(r["seq"]) - 1),
            "T": r["T"], "gtt": gtt, "gtc": r.get("gtc", gtt),
            "freq": r.get("freq", math.inf if car else 1.0),
            "M": r.get("M", math.nan),
            "first_link": r.get("first_link", -1 if car else r["cand"]),
            "key": r.get("key", f"{r['cand']:06d}"),
            "cities": r.get("cities", "")})
    return pd.DataFrame(out)


def run(rows, order="r3_first", guarantee=True, **changes):
    cands = rows if isinstance(rows, pd.DataFrame) else make(rows)
    return rules.apply_rules(cands, {**BASE, **changes}, order, guarantee)


def status(trace):
    return dict(zip(trace["cand"], trace["status"]))


# --- Small pieces ------------------------------------------------------------
def test_mode_class():
    assert rules.mode_class("RR") == "R"
    assert rules.mode_class("RAB") == "ABR"
    assert rules.mode_class("BRB") == "BR"
    assert rules.mode_class("C") == "C"


def test_class_eps():
    # classes 0 (T 250 and 200) and 1 (T 600)
    eps = rules.class_eps(np.array([0, 0, 1]), np.array([250.0, 200.0, 600.0]),
                          2, BASE)
    assert eps.tolist() == [15, 30]      # 5% of 200 is below the floor; 5% of 600
    rows = [{"cand": 1, "seq": "R", "T": 250}, {"cand": 2, "seq": "RR", "T": 200},
            {"cand": 3, "seq": "B", "T": 600}]
    alts, _ = run(rows)
    assert dict(zip(alts["cls"], alts["eps_T"])) == {"R": 15, "B": 30}


def test_input_checks():
    c = make([{"cand": 1, "seq": "R", "T": 250}])
    with pytest.raises(ValueError):
        rules.apply_rules(c.drop(columns="key"), BASE)
    with pytest.raises(ValueError):
        rules.apply_rules(pd.concat([c, c]), BASE)
    with pytest.raises(ValueError):
        rules.apply_rules(c, BASE, order="november")
    alts, trace = rules.apply_rules(c, BASE)
    assert list(alts.columns) == rules.ALTERNATIVES
    assert list(trace.columns) == rules.TRACE
    empty = rules.apply_rules(c.iloc[:0], BASE)
    assert empty[0].empty and list(empty[0].columns) == rules.ALTERNATIVES
    assert empty[1].empty and list(empty[1].columns) == rules.TRACE
    # without the optional money column
    alts, trace = rules.apply_rules(c.drop(columns="M"), BASE)
    assert alts["alt"].tolist() == [1] and math.isnan(alts["M"][0])


# --- 1. The nine candidates of the 16 Oct note --------------------------------
# path-set-rules-verify_example.py: (mode sequence, T, M); GTC = M + T/3 + 10 n
NINE = {1: ("R", 250, 70), 2: ("RR", 235, 66), 3: ("RR", 215, 70),
        4: ("RR", 220, 72), 5: ("RRR", 205, 65), 6: ("A", 200, 150),
        7: ("B", 420, 30), 8: ("RA", 260, 160), 9: ("C", 270, 55)}
NOTE = dict(EPS_T_ABS=15, EPS_T_REL=0, EPS_M=5, MU=0.5, K_CAP=8, RHO_F=0,
            CROSS_CLASS=True)


def nine():
    rows = []
    for i, (seq, T, M) in NINE.items():
        n = 0 if seq == "C" else len(seq) - 1
        g = M + T / 3 + 10 * n
        rows.append({"cand": i, "seq": seq, "T": T, "M": M, "gtt": g, "gtc": g})
    return rows


@pytest.mark.parametrize("order", ["r3_first", "october"])
def test_nine_candidates(order):
    alts, trace = run(nine(), order, **NOTE)
    assert alts["alt"].tolist() == [9, 3, 1, 7, 6, 8]       # increasing GTC
    st = status(trace)
    assert st[4] == "variant_of:3"
    assert st[2] == "removed_by:3" and st[5] == "removed_by:3"
    assert st[8] == "kept_guarantee"
    assert [st[i] for i in (1, 3, 6, 7, 9)] == ["kept"] * 5
    p3 = alts.set_index("alt").loc[3]
    assert p3["members"] == [3, 4] and p3["n_variants"] == 1
    assert (alts["eps_T"] == 15).all()
    # the numbers of the tests are in the trace
    detail = trace.set_index("cand")["detail"]
    assert detail[5] == "n 1 <= 2; T 215 <= 205 + 15; M 70 <= 65 + 5"
    assert detail[4] == "T 220 - 215 <= 15"
    assert detail[8].startswith("last of class AR; n 1 <= 1; T 215 <= 260 + 0")


def test_nine_candidates_without_guarantee():
    alts, trace = run(nine(), guarantee=False, **NOTE)
    assert alts["alt"].tolist() == [9, 3, 1, 7, 6]
    # P8 is strictly worse than P1, P3 and P6; the lowest GTC is named
    assert status(trace)[8] == "removed_by:3"
    assert trace.set_index("cand")["rule"][8] == "R2-cross"


def test_nine_candidates_cross_class_off():
    alts, _ = run(nine(), **{**NOTE, "CROSS_CLASS": False})
    assert set(alts["alt"]) == {1, 3, 6, 7, 8, 9}
    assert (alts["status"] == "kept").all()


# --- 2. The relation is not transitive; the stable set repairs it -------------
def test_xyz_two_bands():
    x, y, z = ({"cand": i, "seq": "R", "T": T, "M": M}
               for i, (T, M) in enumerate([(300, 60), (320, 50), (340, 40)]))
    alts, trace = run([x, y, z], EPS_T_ABS=10, EPS_T_REL=0, EPS_M=10, RHO_F=0)
    assert set(alts["alt"]) == {0, 2}
    assert status(trace)[1] == "removed_by:0"
    adj = rules.dominance([0, 0, 0], [300, 320, 340], [60, 50, 40], [1, 1, 1],
                          [1, 1, 1], 10, 10, 0)
    assert adj[0, 1] and adj[1, 2] and not adj[0, 2]
    assert adj.sum() == 2
    kept, by = rules.stable_set(adj)
    assert kept.tolist() == [True, False, True] and by.tolist() == [-1, 0, -1]


def test_pqr_chain_of_the_note():
    """p = (120 min, 50), q = (110, 60), r = (100, 70): p removes q, q removes
    r, p does not remove r; "remove everything dominated" would keep p alone."""
    rows = [{"cand": i, "seq": "R", "T": T, "M": M}
            for i, (T, M) in enumerate([(120, 50), (110, 60), (100, 70)])]
    alts, trace = run(rows, EPS_T_ABS=15, EPS_T_REL=0, EPS_M=5, RHO_F=0)
    assert set(alts["alt"]) == {0, 2}
    assert status(trace)[1] == "removed_by:0"


def test_inside_the_band_nothing_is_removed():
    """Equal transfers and clock times at most one band apart: three
    sequences of one class stay three alternatives; one minute more and the
    fastest removes the slowest."""
    seqs = ["RRB", "RBR", "BRR"]
    rows = [{"cand": i, "seq": s, "T": T} for i, (s, T) in
            enumerate(zip(seqs, [200, 210, 215]))]
    alts, _ = run(rows, EPS_T_REL=0, RHO_F=0)
    assert set(alts["alt"]) == {0, 1, 2}
    rows[2]["T"] = 216
    alts, trace = run(rows, EPS_T_REL=0, RHO_F=0)
    assert set(alts["alt"]) == {0, 1} and status(trace)[2] == "removed_by:0"


def test_cycle_raises():
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    with pytest.raises(RuntimeError):
        rules.stable_set(cycle)


# --- 3. Only time and money may carry a band ---------------------------------
@pytest.mark.parametrize("extra", [{"EPS_N": 1}, {"EPS_DEP": 15},
                                   {"EPS_TRANSFER_TIME": 5}])
def test_third_band_refused(extra):
    with pytest.raises(ValueError, match="only time and money"):
        rules.check_params({**BASE, **extra})
    with pytest.raises(ValueError, match="only time and money"):
        run([{"cand": 1, "seq": "R", "T": 250}], **extra)


def test_params_accepted_and_refused():
    rules.check_params(BASE)
    rules.check_params(io.P)
    rules.check_params({**BASE, "EPS_N": 0, "EPS_M": 5})
    for bad in ({"R1_KEY": "chain"}, {"R2_TIME": "gtc"}, {"RHO_F": 2},
                {"EPS_T_ABS": -1}):
        with pytest.raises(ValueError):
            rules.check_params({**BASE, **bad})


# --- 4-5. The frequency gate --------------------------------------------------
def test_daily_direct_against_hourly_one_transfer():
    rows = [{"cand": 1, "seq": "R", "T": 300, "freq": 1},
            {"cand": 2, "seq": "RR", "T": 300, "freq": 16}]
    alts, _ = run(rows, RHO_F=0.5)
    assert set(alts["alt"]) == {1, 2}
    alts, trace = run(rows, RHO_F=0)        # gate off: the direct removes it
    assert set(alts["alt"]) == {1} and status(trace)[2] == "removed_by:1"


def test_rare_fast_against_frequent_slow():
    rows = [{"cand": 1, "seq": "R", "T": 200, "freq": 1},
            {"cand": 2, "seq": "R", "T": 220, "freq": 20}]
    alts, _ = run(rows, RHO_F=0.5)
    assert set(alts["alt"]) == {1, 2}
    alts, _ = run(rows, RHO_F=0)
    assert set(alts["alt"]) == {1}


def test_rare_fast_leader_is_not_the_representative():
    """Inside the band the two are one alternative; the frequent member
    represents it and the trace says why the faster one does not."""
    rows = [{"cand": 1, "seq": "R", "T": 200, "freq": 1},
            {"cand": 2, "seq": "R", "T": 210, "freq": 20}]
    alts, trace = run(rows, RHO_F=0.5)
    assert alts["alt"].tolist() == [2] and alts["T"].tolist() == [210]
    row = trace.set_index("cand").loc[1]
    assert row["status"] == "variant_of:2" and row["by"] == 1
    assert row["detail"] == "T 200 - 200 <= 15; freq 1 < 0.5 x 20"


def test_pooled_frequency_counts_a_first_link_once():
    rows = [{"cand": 1, "seq": "RR", "T": 300, "freq": 4, "first_link": 10},
            {"cand": 2, "seq": "RR", "T": 305, "freq": 6, "first_link": 10},
            {"cand": 3, "seq": "RR", "T": 310, "freq": 2, "first_link": 11}]
    alts, _ = run(rows)
    assert alts["alt"].tolist() == [1] and alts["freq"].tolist() == [4]
    assert alts["freq_pooled"].tolist() == [8]      # 6 + 2, not 12
    # the pooled value is what a remover must match
    rows.append({"cand": 4, "seq": "R", "T": 300, "freq": 3})
    alts, _ = run(rows)
    assert set(alts["alt"]) == {1, 4}               # 3 < 0.5 x 8
    rows[-1]["freq"] = 4
    alts, _ = run(rows)
    assert set(alts["alt"]) == {4}


# --- 6. R3 on both quantities -------------------------------------------------
def test_r3_needs_both():
    weekly = {"cand": 1, "seq": "R", "T": 250, "gtt": 250, "gtc": 490, "freq": 1 / 7}
    hourly = {"cand": 2, "seq": "RRR", "T": 330, "gtt": 380, "gtc": 399, "freq": 16}
    cut, b_gtt, b_gtc = rules.r3_cut(np.array([0, 0]), np.array([250.0, 380.0]),
                                     np.array([490.0, 399.0]), 0.5)
    assert not cut.any()
    assert b_gtt.tolist() == [375, 375] and b_gtc.tolist() == [598.5, 598.5]
    alts, trace = run([weekly, hourly])
    assert set(alts["alt"]) == {1, 2}
    # outside the margin on both: cut
    slow = {"cand": 3, "seq": "RRR", "T": 400, "gtt": 380, "gtc": 600, "freq": 16}
    _, trace = run([weekly, hourly, slow])
    st = trace.set_index("cand")
    assert st["status"][3] == "cut_r3" and st["alt"][3] == -1
    assert st["detail"][3] == "gtt 380 > 375; gtc 600 > 598.5 (1.5 x class minimum)"
    assert st["status"][2] == "kept"
    # the October order measures gtc alone: a weekly direct whose hidden wait
    # puts it above 1.5 x 399 is cut there and kept here
    weekly_far = {**weekly, "gtc": 700}
    alts, trace = run([weekly_far, hourly], order="october")
    assert status(trace)[1] == "cut_r3" and alts["alt"].tolist() == [2]
    alts, _ = run([weekly_far, hourly], order="r3_first")
    assert set(alts["alt"]) == {1, 2}


# --- 7. Oded's case -----------------------------------------------------------
def test_one_transfer_twenty_minutes_faster():
    direct = {"cand": 1, "seq": "R", "T": 250}
    alts, _ = run([direct, {"cand": 2, "seq": "RR", "T": 230}])
    assert set(alts["alt"]) == {1, 2}
    alts, trace = run([direct, {"cand": 2, "seq": "RR", "T": 240}])
    assert set(alts["alt"]) == {1}
    row = trace.set_index("cand").loc[2]
    assert row["status"] == "removed_by:1" and row["rule"] == "R2"
    assert row["detail"] == "n 0 <= 1; T 250 <= 240 + 15; freq 1 >= 0.5 x 1"


def test_weighted_time_changes_the_case():
    """R2_TIME = weighted compares gtt: with the 20 clock minutes worth only
    10 weighted minutes the one-transfer path is removed."""
    rows = [{"cand": 1, "seq": "R", "T": 250, "gtt": 300},
            {"cand": 2, "seq": "RR", "T": 230, "gtt": 290}]
    assert set(run(rows)[0]["alt"]) == {1, 2}
    assert set(run(rows, R2_TIME="weighted")[0]["alt"]) == {1}


# --- 8. A faster variant is not hidden ---------------------------------------
def test_faster_variant_represents():
    rows = [{"cand": 1, "seq": "R", "T": 250},
            {"cand": 2, "seq": "RR", "T": 235, "cities": "a-b-c"},
            {"cand": 3, "seq": "RR", "T": 221, "cities": "a-d-c"}]
    alts, trace = run(rows)
    a = alts.set_index("alt")
    assert set(a.index) == {1, 3}
    assert a["T"][3] == 221 and a["members"][3] == [3, 2]
    assert a["cities"][3] == "a-d-c"
    assert status(trace)[2] == "variant_of:3"


def test_r1_key_class():
    """Keyed on the class, coach-rail and rail-coach five minutes apart are
    one alternative; a direct path still never merges with a transfer path."""
    rows = [{"cand": 1, "seq": "RB", "T": 300}, {"cand": 2, "seq": "BR", "T": 305},
            {"cand": 3, "seq": "R", "T": 250}, {"cand": 4, "seq": "RR", "T": 255}]
    alts, _ = run(rows)
    assert set(alts["alt"]) == {1, 2, 3}
    alts, trace = run(rows, R1_KEY="class")
    assert set(alts["alt"]) == {1, 3}
    st = status(trace)
    assert st[2] == "variant_of:1" and st[4] == "removed_by:3"


# --- 9. Properties on random instances ---------------------------------------
SEQS = ["R", "RR", "RRR", "B", "RB", "BR", "RBR", "BRR", "A", "AA", "RA",
        "AR", "RAR", "BA", "RAB"]


def random_cands(rng, money):
    """money: 'none' | 'all' | 'some' (M known for about half)."""
    k = int(rng.integers(3, 45))
    rows = []
    for i in range(k):
        seq = SEQS[rng.integers(len(SEQS))]
        T = float(200 + rng.integers(0, 150))
        f = float(rng.choice([1 / 7, 1, 2, 4, 8, 16]))
        gtt = round(T * float(rng.uniform(1.0, 1.7)), 1)
        M = float(40 + rng.integers(0, 60))
        if money == "none" or (money == "some" and rng.random() < 0.5):
            M = math.nan
        rows.append({"cand": 100 + i, "seq": seq, "T": T, "gtt": gtt,
                     "gtc": gtt + min(240.0, 300 / f), "freq": f, "M": M,
                     "first_link": int(rng.integers(0, 8)),
                     "key": f"{rng.integers(10**6):06d}-{i}"})
    if rng.random() < 0.5:
        rows.append({"cand": 99, "seq": "C", "T": 320.0, "M": 80.0 if
                     money == "all" else math.nan})
    return make(rows)


def removes(p, q, eps_T, eps_M, rho, x):
    """R2 as the task states it, one pair at a time (independent of the
    vectorised relation)."""
    if p["n"] > q["n"] or p[x] > q[x] + eps_T:
        return False
    better = p["n"] < q["n"] or p[x] < q[x] - eps_T
    if eps_M is not None:
        if p["M"] > q["M"] + eps_M:
            return False
        better = better or p["M"] < q["M"] - eps_M
    return better and (rho == 0 or p["freq"] >= rho * q["pooled"])


def slow_r3_r1(cands, params):
    """R3 and R1 as the task states them, one candidate at a time: the ids
    cut, the groups {representative: attributes and members} and the time
    band per class."""
    rows = cands.to_dict("records")
    low = {}
    for r in rows:
        r["cls"] = "".join(sorted(set(r["seq"])))
        g, h = low.get(r["cls"], (math.inf, math.inf))
        low[r["cls"]] = (min(g, r["gtt"]), min(h, r["gtc"]))
    f = 1 + params["MU"]
    cut = {r["cand"] for r in rows if r["gtt"] > f * low[r["cls"]][0]
           and r["gtc"] > f * low[r["cls"]][1]}
    live = [r for r in rows if r["cand"] not in cut]
    best = {}
    for r in live:
        best[r["cls"]] = min(best.get(r["cls"], math.inf), r["T"])
    eps = {k: max(params["EPS_T_ABS"], params["EPS_T_REL"] * v)
           for k, v in best.items()}
    rho, eps_M = params["RHO_F"], params["EPS_M"]
    groups = {}
    for r in sorted(live, key=lambda r: (r["T"], r["gtt"], r["key"])):
        key = r["seq"] if params["R1_KEY"] == "sequence" else (r["cls"], r["n"])
        for g in groups.setdefault(key, []):
            near = r["T"] - g[0]["T"] <= eps[r["cls"]]
            if eps_M is not None and not (math.isnan(r["M"]) or math.isnan(g[0]["M"])):
                near = near and abs(r["M"] - g[0]["M"]) <= eps_M
            if near:
                g.append(r)
                break
        else:
            groups[key].append([r])
    alts = {}
    for g in itertools.chain.from_iterable(groups.values()):
        top = max(m["freq"] for m in g)
        rep = next(m for m in g if rho == 0 or m["freq"] >= rho * top)
        per_link = {}
        for m in g:
            per_link[m["first_link"]] = max(per_link.get(m["first_link"], 0.0),
                                            m["freq"])
        alts[rep["cand"]] = {**rep, "pooled": sum(per_link.values()),
                             "members": [m["cand"] for m in g]}
    return cut, alts, eps


SETTINGS = [
    dict(),                                              # the defaults
    dict(RHO_F=1.0),
    dict(RHO_F=0.33, EPS_T_REL=0.0),
    dict(EPS_M=5),
    dict(EPS_M=10, R2_TIME="weighted"),
    dict(R1_KEY="class", EPS_M=5),
    dict(RHO_F=0.0, EPS_M=5),
]


@pytest.mark.parametrize("money", ["none", "all", "some"])
@pytest.mark.parametrize("setting", range(len(SETTINGS)))
def test_properties_of_the_kept_set(money, setting):
    params = {**BASE, "K_CAP": 1000, **SETTINGS[setting]}
    x = rules.R2_TIMES[params["R2_TIME"]]
    rng = np.random.default_rng(20261113 + setting)
    for _ in range(30):
        cands = random_cands(rng, money)
        alts, trace = rules.apply_rules(cands, params)       # never raises
        assert trace["cand"].tolist() == sorted(cands["cand"])
        st = status(trace)
        # R3 and R1 against the slow version
        cut, built, eps_T = slow_r3_r1(cands, params)
        assert cut == {c for c, s in st.items() if s == "cut_r3"}
        of = dict(zip(trace["cand"], trace["alt"]))
        for a, v in built.items():
            assert all(of[m] == a for m in v["members"])
            assert all(st[m] == f"variant_of:{a}" for m in v["members"] if m != a)
        assert all(of[c] == -1 for c in cut)
        assert len(cut) + sum(len(v["members"]) for v in built.values()) == len(st)
        # the kept set, with the representative's attributes
        kept = {a for a in built if st[a] == "kept"}
        assert alts["alt"].tolist() == sorted(
            kept, key=lambda a: (built[a]["gtc"], built[a]["key"]))
        for row in alts.to_dict("records"):
            want = built[row["alt"]]
            assert row["members"] == want["members"]
            assert row["n_variants"] == len(want["members"]) - 1
            assert row["freq_pooled"] == pytest.approx(want["pooled"])
            assert row["eps_T"] == eps_T[row["cls"]]
            for k in ("seq", "n", "T", "gtt", "gtc", "freq", "first_link", "key"):
                assert row[k] == want[k]
        # money is compared in a class when known for all its alternatives
        eps_M = {k: params["EPS_M"] if not any(
            math.isnan(v["M"]) for v in built.values() if v["cls"] == k) else None
            for k in eps_T}
        args = lambda k: (eps_T[k], eps_M[k], params["RHO_F"], x)  # noqa: E731
        # no kept alternative is removed by a kept one
        for p, q in itertools.permutations(kept, 2):
            if built[p]["cls"] == built[q]["cls"]:
                assert not removes(built[p], built[q], *args(built[p]["cls"]))
        # every other alternative names a kept remover that does remove it
        for q in set(built) - kept:
            assert st[q].startswith("removed_by:")
            p = int(st[q].split(":")[1])
            assert p in kept and built[p]["cls"] == built[q]["cls"]
            assert removes(built[p], built[q], *args(built[q]["cls"]))
        # the order of the input rows does not matter
        shuffled = cands.sample(frac=1, random_state=int(rng.integers(10**6)))
        alts2, trace2 = rules.apply_rules(shuffled, params)
        pd.testing.assert_frame_equal(alts, alts2)
        pd.testing.assert_frame_equal(trace, trace2)


@pytest.mark.parametrize("order", ["r3_first", "october"])
def test_properties_with_cross_class_and_cap(order):
    params = {**BASE, "CROSS_CLASS": True, "K_CAP": 4, "K_EXTRA": 1, "EPS_M": 5}
    rng = np.random.default_rng(7)
    seen = set()
    for i in range(90):
        cands = random_cands(rng, ["none", "all", "some"][i % 3])
        classes = set(cands["seq"].map(rules.mode_class))
        alts, trace = rules.apply_rules(cands, params, order)
        st, rule = status(trace), dict(zip(trace["cand"], trace["rule"]))
        seen |= {s.split(":")[0] for s in st.values()}
        assert set(alts["cls"]) == classes                 # R4
        assert len(alts) <= max(4, len(classes) + 1)       # R5
        assert set(alts["status"]) <= {"kept", "kept_guarantee"}
        assert set(alts["alt"]) == {c for c, s in st.items() if s.startswith("kept")}
        # the remover named by a comparison survived that comparison
        for q, s in st.items():
            if s.startswith("removed_by:"):
                p = int(s.split(":")[1])
                assert not (st[p].startswith("removed_by:") and rule[p] == rule[q])
        without, _ = rules.apply_rules(cands, params, order, guarantee=False)
        assert (without["status"] == "kept").all()
        assert set(without["cls"]) <= classes
        shuffled = cands.sample(frac=1, random_state=i)
        alts2, trace2 = rules.apply_rules(shuffled, params, order)
        pd.testing.assert_frame_equal(alts, alts2)
        pd.testing.assert_frame_equal(trace, trace2)
    assert {"kept", "kept_guarantee", "variant_of", "removed_by", "capped"} <= seen
    if order == "october":
        assert "cut_r3" in seen


def kernels(adj):
    """All independent and absorbing sets, by enumeration (all_kernels of
    path-set-rules-verify_r2.py)."""
    k = len(adj)
    out = []
    for mask in range(1 << k):
        K = [v for v in range(k) if mask >> v & 1]
        if any(adj[u][v] for u in K for v in K):
            continue
        if all(any(adj[u][v] for u in K) for v in range(k) if v not in K):
            out.append(set(K))
    return out


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eps_M", [None, 5])
def test_stable_set_is_the_unique_kernel(rho, eps_M):
    """With the gate on or off, one or two bands and money known for all, the
    relation has no cycle: elimination never raises and returns the one set
    that is independent and absorbing."""
    rng = np.random.default_rng(16)
    for _ in range(400):
        k = int(rng.integers(2, 9))
        n = rng.integers(0, 3, k)
        X = 240 + rng.integers(-20, 21, k)
        M = 60 + rng.integers(-7, 8, k).astype(float)
        freq = rng.choice([0.5, 1, 2, 4, 8], k)
        pooled = freq * rng.choice([1, 1.5, 2], k)
        adj = rules.dominance(n, X, M, freq, pooled, 15, eps_M, rho)
        for p, q in itertools.permutations(range(k), 2):
            a = {"n": n[p], "T": X[p], "M": M[p], "freq": freq[p]}
            b = {"n": n[q], "T": X[q], "M": M[q], "pooled": pooled[q]}
            assert adj[p, q] == removes(a, b, 15, eps_M, rho, "T")
        assert not adj.diagonal().any()
        kept, by = rules.stable_set(adj)
        assert kernels(adj.tolist()) == [set(np.flatnonzero(kept).tolist())]
        assert all(adj[by[q], q] and kept[by[q]] for q in np.flatnonzero(~kept))


def test_money_known_for_some_only():
    """With M missing for one alternative, the pair-by-pair relation has a
    cycle a -> b -> c -> d -> u -> a and no stable set; apply_rules then
    compares the class on time alone and does not raise."""
    T = [345, 330, 315, 300, 322]
    M = [70, 80, 90, 100, math.nan]
    adj = rules.dominance([2] * 5, T, M, [1] * 5, [1] * 5, 15, 5, 0)
    assert all(adj[i, (i + 1) % 5] for i in range(5))
    with pytest.raises(RuntimeError, match="cycle"):
        rules.stable_set(adj)
    seqs = ["RRB", "RBR", "BRR", "RBB", "BRB"]
    rows = [{"cand": i, "seq": s, "T": t, "M": m}
            for i, (s, t, m) in enumerate(zip(seqs, T, M))]
    alts, trace = run(rows, EPS_T_REL=0, EPS_M=5, RHO_F=0)
    assert set(alts["alt"]) == {2, 3}                 # T 315 and 300
    assert "M" not in trace.set_index("cand")["detail"][0]


# --- 10. The cap --------------------------------------------------------------
EIGHT = ["A", "B", "R", "AB", "AR", "BR", "RAB", "C"]


def eight_classes():
    rows = [{"cand": i, "seq": s, "T": 300 + 10 * i, "gtc": 400 + 10 * i}
            for i, s in enumerate(EIGHT)]
    direct = rows[2]
    rows.append({"cand": 8, "seq": "RR", "T": direct["T"] - 20, "gtc": 900})
    return rows


def test_cap_leaves_room_beside_eight_classes():
    alts, trace = run(eight_classes())
    assert len(alts) == 9 and "capped" not in set(trace["status"])
    assert status(trace)[8] == "kept"
    # without the floor the one-transfer rail path, last by gtc, is dropped
    alts, trace = run(eight_classes(), K_EXTRA=0)
    row = trace.set_index("cand").loc[8]
    assert row["status"] == "capped" and len(alts) == 8
    assert row["detail"] == "gtc 900; rank 9 of 9; cap 8"


def test_cap_never_drops_the_last_of_a_class():
    drop = rules.cap_drop(["R", "A", "R", "B", "R", "A"], 3)
    assert drop.tolist() == [False, False, True, False, True, True]
    drop = rules.cap_drop(["R", "A", "B", "C"], 2)
    assert not drop.any()
    drop = rules.cap_drop(["R", "R", "R"], 5)
    assert not drop.any()
