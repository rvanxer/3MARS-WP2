"""Stage C/D report (pathset/stagecd.py) on small synthetic tables: R0
intervals, the search log of the shards, the console log, path metrics, set
sizes and the cap, common periods of links, overlaps, R3 on a subset (the
reason one search at MU = 1.0 serves smaller MU), the per-pair sensitivity,
the sentences computed from the sensitivity tables, and the checks of the
open items and of the review items. No search runs here."""

import datetime as dt
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import cost as C  # noqa: E402
from pathset import io, rules  # noqa: E402
from pathset import stagecd as S  # noqa: E402
from pathset import view as V  # noqa: E402

BASE = {**io.P, "EPS_T_ABS": 15, "EPS_T_REL": 0.05, "EPS_M": None, "MU": 0.5,
        "K_CAP": 8, "K_EXTRA": 3, "RHO_F": 0.5, "CROSS_CLASS": False,
        "R1_KEY": "sequence", "R2_TIME": "clock", "R2_GATE": "top", "R4_GUARANTEE": True}


# --- Formatting --------------------------------------------------------------
def test_formatting():
    assert S.n_(1234567) == "1,234,567" and S.n_(math.nan) == ""
    assert S.f_(1234.56, 1) == "1,234.6" and S.f_(np.inf) == "inf"
    assert S.pct(1, 3) == "33.3%" and S.pct(1, 0) == ""
    assert S.hm(61) == "1:01" and S.hm(59.6) == "1:00" and S.hm(1439.5) == "24:00"
    assert S.hm(np.nan) == ""
    assert S.mean_sd([1, 2, 3]) == "2 (1)" and S.mean_sd([]) == ""
    t = S.md(pd.DataFrame({"a": [1000, 2], "b": [1.25, np.nan], "c": ["x|y", "z"]}))
    assert t.splitlines()[2] == "| 1,000 | 1.2 | x\\|y |"
    assert t.splitlines()[3] == "| 2 |  | z |"


# --- R0 intervals and sequences ---------------------------------------------
def test_r0_interval():
    lo, hi = S.r0_interval([10, 5, 7], [3, 0, 7])
    assert lo.tolist() == [7, 5, 0] and hi.tolist() == [10, 5, 7]
    with pytest.raises(ValueError):
        S.r0_interval([3], [4])
    with pytest.raises(ValueError):
        S.r0_interval([3], [-1])


def test_seq_table_sums_and_marks_exact_counts():
    sc = pd.DataFrame({
        "o": [1, 1, 1, 1, 2, 2, 2, 2],
        "seq": ["R", "RR", "RRR", "C", "R", "RR", "RRR", "C"],
        "n_r0": [5, 20, 100, 1, 6, 30, 200, 1],
        "n_r0_bound": [0, 0, 60, 0, 0, 0, 150, 0],
        "n_r3": [5, 10, 20, 1, 6, 12, 25, 1],
        "n_kept": [2, 3, 1, 1, 2, 2, 1, 1]})
    t = S.seq_table(sc)
    assert t["seq"].tolist() == ["C", "R", "RR", "RRR"]
    assert t["links"].tolist() == [0, 1, 2, 3]
    r = t.set_index("seq")
    assert r.loc["RRR", "n_r0"] == 300 and r.loc["RRR", "r0_low"] == 90
    assert r.loc["RRR", "r0_high"] == 300 and not r.loc["RRR", "exact"]
    assert r.loc["RR", "exact"] and r.loc["RR", "r0_low"] == r.loc["RR", "r0_high"] == 50
    assert r.loc["C", "n_kept"] == 2


# --- Search log --------------------------------------------------------------
def _pass(s, ch3, skip, calls, reused=0, ev=0, r0b=0, cache=0, cleared=0):
    return {"s": s, "cache_entries": cache, "chains1": 3, "chains2": 30, "chains3": ch3,
            "evaluated": ev, "triple_calls": calls, "triple_reused": reused,
            "chains3_skipped": skip, "r0_by_bounds": r0b, "written": 0, "triple_rows": 0,
            "cache_cleared": cleared}


def test_search_stats_from_attrs():
    attrs = {"search": {"origin": 7, "blocks": 2, "buffer_mb": 12.5, "rules_s": 10.0,
                        "total_s": 100.0, "rss_start_mb": 1500.0, "peak_mb": 3000.0,
                        "pass1": _pass(20.0, 1000, 900, 50, cache=50),
                        "pass2": _pass(40.0, 1000, 600, 250, reused=30, ev=700, r0b=900,
                                       cache=300, cleared=1),
                        "trace_pass2": _pass(5.0, 100, 10, 90, cache=90)},
             "param_hash": "abc"}
    s = S.search_stats(attrs)
    assert s["origin"] == 7 and s["blocks"] == 2
    assert s["other_s"] == pytest.approx(100 - 20 - 40 - 5 - 10)
    assert s["skip_share1"] == pytest.approx(0.9) and s["skip_share2"] == pytest.approx(0.6)
    assert s["calls1"] == 50 and s["calls2"] == 250 and s["calls_trace"] == 90
    assert s["calls_per_s"] == pytest.approx(300 / 60)
    assert s["reused2"] == 30 and s["evaluated2"] == 700 and s["r0_by_bounds"] == 900
    assert s["cache_max"] == 300 and s["cache_cleared"] == 1
    # without a trace pass, and the attrs dict given as the inner dict
    inner = dict(attrs["search"])
    del inner["trace_pass2"]
    s2 = S.search_stats(inner)
    assert s2["trace_s"] == 0.0 and s2["other_s"] == pytest.approx(30.0)
    t = S.timing_table({9: {"search": {**inner, "origin": 9}}, 7: attrs})
    assert t["origin"].tolist() == [7, 9]


def test_parse_log_crosses_midnight():
    text = "\n".join([
        "23:36:17 enumerate: 3 origins to run, 0 finished, 4 workers, MU 0.5, "
        "PASSED_SHARE 0.5, hash 467483438c68c5c1",
        "23:37:28 saved paths/alternatives/o002: 2,919 rows",
        "23:37:28 saved paths/funnel/o002: 383 rows",
        "23:37:28 o 2: r0 789,108 (by bounds 297,849) ...",
        "not a log line",
        "00:10:00 saved paths/funnel/o001: 383 rows",
        "00:10:00 o 1: r0 ...",
        "01:11:03 saved paths/funnel/o000: 383 rows",
        "01:11:03 o 0: r0 ..."])
    r = S.parse_log(text, dt.date(2026, 11, 13))
    assert r["start"] == dt.datetime(2026, 11, 13, 23, 36, 17)
    assert r["end"] == dt.datetime(2026, 11, 14, 1, 11, 3)
    assert r["seconds"] == 5686 and r["hash"] == "467483438c68c5c1"
    assert r["workers"] == 4 and r["origins"] == 3 and r["funnels"] == 3
    assert S.parse_log("", dt.date(2026, 1, 1))["start"] is None


def test_parse_log_line_out_of_order_keeps_the_date():
    """A line a few seconds out of order does not move the date; midnight
    does, once."""
    text = "\n".join([
        "23:59:50 enumerate: 2 origins to run, 0 finished, 2 workers, MU 0.5, "
        "PASSED_SHARE 0.5, hash abc",
        "23:59:58 saved paths/funnel/o001: 383 rows",
        "23:59:57 saved paths/funnel/o002: 383 rows",
        "00:00:05 saved paths/funnel/o003: 383 rows"])
    r = S.parse_log(text, dt.date(2026, 11, 13))
    assert r["end"] == dt.datetime(2026, 11, 14, 0, 0, 5)
    assert r["seconds"] == 15


def test_wall_text_never_prints_sixty_minutes():
    assert S.wall_text(5686) == "1 h 35 min"
    assert S.wall_text(3599.6) == "1 h 0 min"
    assert S.wall_text(3570) == "1 h 0 min"
    assert S.num(1248) == "1,248" and S.num(97) == "97"


def test_run_records_and_record_of():
    recs = [{"step": "enumerate", "time": "a", "info": {"origins_run": [1, 2], "combined": {}}},
            {"step": "stage-b", "time": "b", "info": {}},
            {"step": "enumerate", "time": "c", "info": {"origins_run": [3], "combined": {}}},
            {"step": "enumerate", "time": "d", "info": {"origins_run": [], "combined": {"x": 1}}}]
    run, comb = S.run_records(recs)
    assert run["time"] == "c" and comb["time"] == "d"
    assert S.record_of(recs, "enumerate", [2, 1])["time"] == "a"
    assert S.record_of(recs, "stage-b")["time"] == "b"
    assert S.run_records(recs[:2])[1] is None
    assert S.run_records([]) == (None, None)


def test_run_hash_ignores_unhashed_keys(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"1234")
    p = {"MU": 0.5, "WORKERS": 4, "SERVICE_START": dt.date(2023, 1, 1)}
    h = S.run_hash(p, True, [f])
    assert len(h) == 16
    assert S.run_hash({**p, "WORKERS": 6}, True, [f]) == h
    assert S.run_hash({**p, "MU": 0.25}, True, [f]) != h
    assert S.run_hash(p, False, [f]) != h
    # dates as written to the manifest (strings) hash alike
    assert S.run_hash({**p, "SERVICE_START": "2023-01-01"}, True, [f]) == h


# --- Path metrics ------------------------------------------------------------
LINKS = pd.DataFrame({"link": [0, 1, 2, 3], "src": [0, 1, 2, 3], "trg": [1, 2, 3, 0],
                      "mode": ["A", "R", "B", "R"], "ride": [60.0, 120.0, 200.0, 30.0],
                      "km": [500.0, 300.0, 250.0, 40.0],
                      "first_day": [0, 100, 300, 100], "last_day": [200, 400, 500, 100]})


def paths(rows):
    out = []
    for r in rows:
        l = list(r["links"]) + [-1] * (3 - len(r["links"]))
        out.append({"o": r.get("o", 1), "d": r.get("d", 2), "seq": r["seq"],
                    "l1": l[0], "l2": l[1], "l3": l[2], "acc": r.get("acc", 10.0),
                    "egr": r.get("egr", 20.0), "basis": r.get("basis", "D"),
                    "cls": rules.mode_class(r["seq"]) if r["seq"] != "C" else "C",
                    "key": r.get("key", "-".join(map(str, r["links"])) or "car"),
                    "h_last": r.get("h_last", -1), "T": r.get("T", 100.0),
                    "rank": r.get("rank", 1)})
    return pd.DataFrame(out)


def test_path_metrics_rides_by_mode_and_km():
    p = paths([{"seq": "A", "links": [0]}, {"seq": "AR", "links": [0, 1]},
               {"seq": "RBR", "links": [3, 2, 1]}, {"seq": "C", "links": [], "acc": 0.0}])
    m = S.path_metrics(p, LINKS)
    assert m["ride_A"].tolist() == [60.0, 60.0, 0.0, 0.0]
    assert m["ride_R"].tolist() == [0.0, 120.0, 150.0, 0.0]
    assert m["ride_B"].tolist() == [0.0, 0.0, 200.0, 0.0]
    assert m["km"].tolist()[:3] == [500.0, 800.0, 590.0] and np.isnan(m["km"].iloc[3])
    assert m["acc"].tolist() == [10.0, 10.0, 10.0, 0.0]
    t = S.metrics_by_seq(m, ["A", "AR", "BB"])
    assert t["rows"].tolist() == [1, 1, 0]
    assert t.loc[0, "ride_A"] == "60 (0)" and t.loc[2, "km"] == ""
    with pytest.raises(ValueError):                      # ids must be 0 .. L-1
        S.path_metrics(p, LINKS.assign(link=[0, 1, 2, 5]))


# --- Set sizes and the cap ---------------------------------------------------
def test_set_size_table_and_cap():
    f = pd.DataFrame({"o": 1, "d": range(6),
                      "n_classes": [2, 5, 5, 6, 8, 8],
                      "n_kept": [2, 8, 6, 9, 11, 10],
                      "n_capped": [0, 3, 0, 1, 4, 0]})
    assert S.cap_of([2, 5, 6, 8], BASE).tolist() == [8, 8, 9, 11]
    t = S.set_size_table(f, BASE).set_index("n_classes")
    assert t.loc[5, "pairs"] == 2 and t.loc[5, "cap"] == 8
    assert t.loc[5, "at_cap"] == 1 and t.loc[5, "pairs_capped"] == 1 and t.loc[5, "capped"] == 3
    assert t.loc[8, "cap"] == 11 and t.loc[8, "at_cap"] == 1 and t.loc[8, "max"] == 11
    assert t.loc[8, "mean"] == pytest.approx(10.5)
    d = S.size_distribution(f)
    assert d["alternatives"].tolist() == [2, 6, 8, 9, 10, 11] and d["pairs"].sum() == 6


def test_cap_detail_from_no_cap_and_default():
    """Ten alternatives before the cap in six classes: cap 9; rules.cap_drop
    drops rank 8 (B, not the last of its class) and keeps rank 10 (AR, the
    last of its class) beyond the cap."""
    cls = ["C", "R", "R", "A", "A", "A", "B", "B", "AB", "AR"]
    nc = pd.DataFrame({"d": 1, "rank": range(1, 11), "key": [f"k{i}" for i in range(10)],
                       "cls": cls})
    drop = rules.cap_drop(cls, int(S.cap_of([6], BASE)[0]))
    assert drop.tolist() == [False] * 7 + [True, False, False]
    c = S.cap_detail(nc[~drop], nc, BASE)
    assert c["dropped"] == 1 and c["dropped_by_class"] == {"B": 1}
    assert c["protected"] == 1 and c["protected_by_class"] == {"AR": 1}
    assert c["pairs_protected"] == 1
    # no cap at work: nothing dropped, nothing protected
    c2 = S.cap_detail(nc.iloc[:5], nc.iloc[:5], BASE)
    assert c2["dropped"] == 0 and c2["protected"] == 0


def test_class_extras():
    a = pd.DataFrame({"o": [1, 1, 1, 1, 2, 2], "d": [2, 2, 2, 2, 3, 3],
                      "cls": ["R", "R", "A", "C", "R", "C"]})
    e = S.class_extras(a)
    assert e["pairs"] == 2 and e["alternatives"] == 6 and e["cells"] == 5 and e["extras"] == 1
    pc = e["per_class"].set_index("cls")
    assert pc.loc["R", "cells"] == 2 and pc.loc["R", "multi"] == 1 and pc.loc["R", "alternatives"] == 3
    assert pc.loc["C", "multi"] == 0


# --- Common period of the links ----------------------------------------------
def test_common_period_and_basis_table():
    p = paths([{"seq": "AR", "links": [0, 1], "basis": "T"},      # [100, 200]: common
               {"seq": "AB", "links": [0, 2], "basis": "T"},      # 300 > 200: disjoint
               {"seq": "RR", "links": [1, 3], "basis": "T"},      # one shared day (100)
               {"seq": "RBR", "links": [3, 2, 1], "basis": "D"},  # 300 > 100: disjoint
               {"seq": "C", "links": [], "basis": ""}])
    cp = S.common_period(p, LINKS)
    assert cp["disjoint"].tolist() == [False, True, False, True, False]
    assert cp["first"].tolist()[:4] == [100, 300, 100, 300] and cp["last"].tolist()[:4] == [200, 200, 100, 100]
    t = S.basis_table(p, cp["disjoint"].to_numpy()).set_index("cls")
    assert t.loc["AB", "basis_T"] == 1 and t.loc["AB", "T_disjoint"] == 1
    assert t.loc["BR", "basis_D"] == 1 and t.loc["BR", "T_disjoint"] == 0   # disjoint but D
    assert t.loc["R", "T_disjoint"] == 0 and t.loc["C", "basis_T"] == 0


# --- Overlap with the default ------------------------------------------------
def test_overlap_on_destination_and_key():
    base = pd.DataFrame({"d": [1, 1, 2], "key": ["5", "6-7", "5"]})
    other = pd.DataFrame({"d": [1, 2, 2], "key": ["5", "6-7", "5"]})
    ov = S.overlap(base, other)
    assert ov == {"n_default": 3, "n_same": 2, "share": pytest.approx(2 / 3)}
    assert math.isnan(S.overlap(base.iloc[:0], other)["share"])


def test_default_mismatch():
    a = pd.DataFrame({"d": [1, 1], "rank": [1, 2], "key": ["car", "5"], "seq": ["C", "R"],
                      "T": np.array([100.0, 120.5], np.float32),
                      "gtt": np.array([100.0, 140.25], np.float32)})
    assert S.default_mismatch(a, a.copy()).empty
    b = a.copy()
    b.loc[1, "T"] = np.float32(120.75)
    m = S.default_mismatch(a, b)
    assert len(m) == 2 and set(m["side"].astype(str)) == {"left_only", "right_only"}


# --- R3 on a subset: one search at MU = 1.0 serves smaller MU -----------------
def synthetic_pair(rng, n=60):
    """Candidates of one pair: several mode classes, the car, float32 values
    as the search stores them."""
    par = C.param_array(BASE)
    seqs = rng.choice(["R", "RR", "B", "BR", "RB", "A", "AR", "ABR"], size=n)
    T = rng.uniform(100, 1400, size=n).round(1)
    gtt = (T * rng.uniform(0.9, 1.6, size=n)).round(1)
    freq = rng.choice([1.0, 2.0, 3.0, 6.0, 12.0], size=n)
    rows = pd.DataFrame({
        "seq": seqs, "n": [len(s) - 1 for s in seqs], "T": T, "gtt": gtt,
        "gtc": gtt + C.hidden_wait(freq, par), "freq": freq,
        "l1": np.arange(100, 100 + n), "l2": -1, "l3": -1,
        "key": [f"k{i:03d}" for i in range(n)], "cities": "",
        "f_fixed": freq / 2, "f_span": freq, "f_first": freq})
    car = pd.DataFrame({"seq": ["C"], "n": [0], "T": [400.0], "gtt": [400.0], "gtc": [400.0],
                        "freq": [np.inf], "l1": [-1], "l2": [-1], "l3": [-1], "key": ["car"],
                        "cities": [""], "f_fixed": [np.inf], "f_span": [np.inf],
                        "f_first": [np.inf]})
    c = pd.concat([car, rows], ignore_index=True)
    for k in ("T", "gtt", "gtc", "freq"):
        c[k] = c[k].astype(np.float32)
    return c


def keys_of(cands, alts):
    k = cands.set_index("cand")["key"]
    return [str(k[a]) for a in alts["alt"]]


@pytest.mark.parametrize("seed", range(6))
def test_mu_filter_on_the_mu1_set_is_exact(seed):
    """R3 at MU <= 1 on the candidates the search keeps at MU = 1.0 gives the
    same alternatives (and order) as on every R0 candidate."""
    rng = np.random.default_rng(seed)
    full = V.number(synthetic_pair(rng))
    sub = V.number(full[S.r3_keep(full, 1.0)].drop(columns="cand"))
    assert len(sub) < len(full)                      # the filter cuts something
    for mu in (0.25, 0.5, 1.0):
        spec = {"params": {"MU": mu}}
        a_full, t_full = V.decide_variant(full, BASE, spec)
        a_sub, t_sub = V.decide_variant(sub, BASE, spec)
        assert keys_of(full, a_full) == keys_of(sub, a_sub)
        assert a_full["T"].tolist() == a_sub["T"].tolist()
        kept_full = int((t_full["status"].astype(str) != "cut_r3").sum())
        kept_sub = int((t_sub["status"].astype(str) != "cut_r3").sum())
        assert kept_full == kept_sub


def test_r3_keep_matches_rules_r3_cut():
    rng = np.random.default_rng(11)
    c = synthetic_pair(rng)
    keep = S.r3_keep(c, 0.5)
    code = pd.factorize(np.array([rules.mode_class(s) for s in c["seq"]]), sort=True)[0]
    cut, _, _ = rules.r3_cut(code, c["gtt"].to_numpy(np.float64), c["gtc"].to_numpy(np.float64), 0.5)
    assert (keep == ~cut).all()
    # the minima of every class are kept
    cls = np.array([rules.mode_class(s) for s in c["seq"]])
    for k in set(cls):
        m = cls == k
        assert keep[m][np.argmin(c["gtt"].to_numpy()[m])]
        assert keep[m][np.argmin(c["gtc"].to_numpy()[m])]


def test_r3_exact_condition():
    c = pd.DataFrame({"seq": ["R", "RR"], "gtt": [100.0, 150.0], "gtc": [120.0, 400.0]})
    assert S.r3_exact(c, 1.0, 0.5)                       # 2 x 100 >= 1.5 x 120
    c2 = c.assign(gtc=[140.0, 400.0])
    assert not S.r3_exact(c2, 1.0, 0.5)                  # 200 < 210
    assert S.r3_exact(c2, 1.0, 0.25)                     # 200 >= 175
    car = pd.DataFrame({"seq": ["C"], "gtt": [300.0], "gtc": [300.0]})
    assert S.r3_exact(car, 1.0, 0.5)


def test_sensitivity_variants_are_valid_specs():
    specs = S.sensitivity_specs()
    assert next(iter(specs)) == "default" and len(specs) == 19
    assert set(S.SENS_NOTES) == set(specs)
    assert not set(S.SENS_LEFT_OUT) & set(specs)
    for name, spec in specs.items():
        assert V.variant_spec(spec) is spec
        rules.check_params({**BASE, **spec.get("params", {})})
        assert spec.get("freq", "path") in ("path", "fixed", "span", "first")


# --- Unserved pairs: graph reach ---------------------------------------------
def test_transfer_csr_and_reach_levels():
    links = pd.DataFrame({"link": [0, 1, 2, 3], "src": [10, 11, 12, 10], "trg": [11, 12, 13, 20]})
    tr = pd.DataFrame({"link_in": [0, 1, 0], "link_out": [1, 2, 2],
                       "basis": ["D", "T", "D"], "n_days": [5, 1, 4]})
    ptr, out = S.transfer_csr(tr, 4, 5)                 # (0, 2) has 4 dates: no chain
    assert ptr.tolist() == [0, 1, 2, 2, 2] and out.tolist() == [1, 2]
    conn = pd.DataFrame({"fua": [1, 2, 3], "hub": [10, 13, 20],
                         "acc_min": [5.0, 5.0, 5.0], "egr_min": [5.0, 5.0, np.nan]})
    assert S.reach_levels(1, 2, conn, links, ptr, out) == [0, 0, 1]
    assert S.reach_levels(1, 3, conn, links, ptr, out) == [0, 0, 0]   # no egress at 20
    assert S.reach_levels(1, 2, conn, links, ptr, out, max_links=2) == [0, 0]


def test_car_coverage():
    car = pd.DataFrame({"o": [1, 1, 2], "d": [2, 3, 1]})
    cov = S.car_coverage(car, [1, 2, 3]).set_index("fua")
    assert cov["out"].tolist() == [2, 1, 0] and cov["in"].tolist() == [1, 1, 1]


# --- Open items --------------------------------------------------------------
def test_passes_through():
    city = np.array([1, 2, 3, 4, 5])                  # hub -> city
    # link 0 passes cities 2 and 9; link 1 passes nothing; link 2 passes city 3
    p_ptr = np.array([0, 2, 2, 3])
    p_city = np.array([2, 9, 3])
    p = pd.DataFrame({"o": [1, 1, 7, 7], "d": [3, 9, 8, 8],
                      "l1": [0, 0, 1, 2], "l2": [-1, -1, 2, 1], "l3": -1,
                      "b1": [0, 0, 0, 0], "a1": [2, 2, 1, 1],
                      "b2": [-1, -1, 1, 2], "a2": [-1, -1, 4, 4], "b3": -1, "a3": -1})
    # 0: o 1, d 3, hub cities {1, 3}; link 0 passes 2 and 9: no hit
    # 1: d 9 is passed: hit
    # 2: link 2 passes 3, the path's cities are {7, 8, 1, 2, 5}: no hit
    # 3: link 2 (first) passes 3, the city of hub 2 where the path boards next: hit
    assert S.passes_through(p, city, p_ptr, p_city).tolist() == [False, True, False, True]


def test_overshoots():
    conn = pd.DataFrame({"fua": [5, 5, 6], "hub": [30, 40, 30], "acc_min": 1.0,
                         "egr_min": [10.0, 10.0, np.nan]})
    p = pd.DataFrame({"d": [5, 5, 6, 5], "h_last": [40, 40, 40, 30],
                      "a1": [30, 31, 30, 30], "b2": [32, 31, 32, -1],
                      "a2": [-1, -1, -1, -1], "b3": [-1, -1, -1, -1]})
    # 0: passes hub 30 (egress to 5) before its last hub 40; 1: no; 2: d 6 has no
    # egress at 30; 3: hub 30 is the last hub itself
    assert S.overshoots(p, conn).tolist() == [True, False, False, False]


def test_r2_details_and_oded_removals():
    det = pd.Series(["n 0 <= 1; T 520 <= 500 + 25.5; freq 4 >= 0.5 x 1",
                     "n 1 <= 1; T 400 <= 420 + 15; freq 2 >= 0.5 x 2",
                     "n 0 <= 2; T 510 <= 500 + 21; freq 1 >= 0.5 x 1",
                     "gtc 807.5; rank 13 of 19; cap 11"])
    d = S.r2_details(det)
    assert d.loc[0].tolist() == [0, 1, 520, 500, 25.5] and d.loc[3].isna().all()
    assert S.oded_removals(d, 15).tolist() == [True, False, False, False]


def test_ftrans_callers(tmp_path):
    (tmp_path / "a.py").write_text("x = PR.ftrans_table(t, l, h, m, days)\n"
                                   "def ftrans_table(t):\n", encoding="utf-8")
    out = S.ftrans_callers(tmp_path, ["a.py", "missing.py"])
    assert out == [("a.py", 1, "x = PR.ftrans_table(t, l, h, m, days)")]


def test_resolve_pairs_uses_local_names():
    fuas = pd.DataFrame({"id": [218, 367, 161], "name": ["München", "Wien", "Köln"]})
    r = S.resolve_pairs(fuas, [("Muenchen", "Wien"), ("Koeln", "Atlantis")])
    assert r[0][:4] == (218, 367, "München", "Wien")
    assert r[1][0] == 161 and r[1][1] is None


# --- Constants ---------------------------------------------------------------
def test_deck_table1_matches_the_slide_totals():
    """Slide 222: '523k direct & 45.95M single-transfer paths'."""
    t = S.DECK_TABLE1
    assert len(t) == 12 and t["seq"].is_unique
    direct = t.loc[t["category"] == "Direct", "paths"].sum()
    one = t.loc[t["category"] != "Direct", "paths"].sum()
    assert round(direct / 1000) == 523 and round(one / 1e4) / 100 == 45.95
    assert (t[["acc_sd", "egr_sd", "km_sd"]] >= 0).all().all()


def test_one_transfer_ratios():
    st = pd.DataFrame({"n_r0": [S.DECK_TABLE1.set_index("seq").loc["AA", "paths"] / 2,
                                S.DECK_TABLE1.set_index("seq").loc["AR", "paths"] * 10]},
                      index=pd.Index(["AA", "AR"], name="seq"))
    r = S.one_transfer_ratios(st).set_index("seq")
    assert len(r) == 9 and r.loc["AA", "ratio"] == pytest.approx(0.5)
    assert r.loc["AR", "ratio"] == pytest.approx(10.0) and np.isnan(r.loc["BB", "ratio"])
    assert S.ratio_text(0.304) == "0.30" and S.ratio_text(25.82) == "25.8"
    assert S.ratio_text(189.26) == "189"


# --- Sensitivity on one pair ---------------------------------------------------
SPECS = {"default": {}, "eps_10": {"params": {"EPS_T_ABS": 10}},
         "no_cap": {"params": {"K_CAP": 1_000_000}},
         "cross_class_no_r4": {"params": {"CROSS_CLASS": True, "R4_GUARANTEE": False}},
         "freq_fixed": {"freq": "fixed"}, "freq_span": {"freq": "span"}}


@pytest.mark.parametrize("seed", range(4))
def test_sensitivity_pair_matches_the_rules(seed):
    """sensitivity_pair: every variant equals view.decide_variant on the same
    candidates; the default's R3 count and R2 removals come from its trace;
    a frequency variant carries r3_exact, the others None."""
    rng = np.random.default_rng(seed)
    cands = V.number(synthetic_pair(rng))
    res = S.sensitivity_pair(cands, BASE, SPECS)
    a0, t0 = V.decide_variant(cands, BASE, {})
    v = res["variants"]
    assert v["default"]["keys"] == keys_of(cands, a0)
    assert res["kept_r3"] == int((t0["status"].astype(str) != "cut_r3").sum())
    rem = t0[t0["rule"].astype(str) == "R2"]
    assert res["r2"]["removals"] == len(rem)
    if len(rem):
        det = S.r2_details(rem["detail"])
        assert res["r2"]["oded"] == int(S.oded_removals(det, BASE["EPS_T_ABS"]).sum())
        assert res["r2"]["eps_max"] == pytest.approx(det["eps"].max())
    assert v["default"]["same_rep"] == v["default"]["found"] == len(a0)
    assert v["no_cap"]["n_capped"] == 0
    for name, spec in SPECS.items():
        a, _ = V.decide_variant(cands, BASE, spec)
        assert v[name]["keys"] == keys_of(cands, a)
        assert (v[name]["exact"] is None) == (name not in S.FREQ_VARIANTS)
    with pytest.raises(ValueError):
        S.sensitivity_pair(cands, BASE, {"eps_10": SPECS["eps_10"]})


def test_run_sensitivity_adds_up_the_pairs(monkeypatch):
    """run_sensitivity on two synthetic destinations (search.candidates
    stubbed): the checks pass against tables built from the same rules, the
    rows add up the pairs, and the size guard refuses a large origin."""
    pairs = {}
    for d, seed in ((1, 21), (2, 22)):
        c = synthetic_pair(np.random.default_rng(seed))
        c.insert(0, "d", d)
        pairs[d] = c
    monkeypatch.setattr(S.search, "candidates",
                        lambda net, o, dests, params, mu: pd.concat(
                            [pairs[d] for d in dests], ignore_index=True))
    links = pd.DataFrame({"link": np.arange(200), "freq": 4.0, "freq_fixed": 2.0,
                          "freq_span": 3.0, "ride": 60.0})
    stored, n3, n_r2 = [], [], 0
    for d, c in pairs.items():
        cands = V.number(V.link_freqs(c, links))
        a, t = V.decide_variant(cands, BASE, {})
        k = keys_of(cands, a)
        stored.append(pd.DataFrame({"d": d, "rank": np.arange(1, len(a) + 1), "key": k,
                                    "seq": a["seq"].to_numpy(), "T": a["T"].to_numpy(),
                                    "gtt": a["gtt"].to_numpy()}))
        n3.append((0, d, int((t["status"].astype(str) != "cut_r3").sum())))
        n_r2 += int((t["rule"].astype(str) == "R2").sum())
    stored = pd.concat(stored, ignore_index=True)
    funnel_o = pd.DataFrame(n3, columns=["o", "d", "n_r3"])

    class Net:
        fuas = np.array([0, 1, 2])
    res = S.run_sensitivity(Net(), 0, BASE, links, stored, funnel_o, specs=SPECS,
                            log=lambda m: None)
    run = res["run"]
    assert run["default_check"] == "pass" and run["r3_check"] == "pass"
    assert run["pairs"] == 2 and run["r2_removals"] == n_r2
    rows = pd.DataFrame(res["rows"]).set_index("variant")
    assert list(rows.index) == list(SPECS)
    assert rows.loc["default", "n_alt"] == len(stored)
    assert rows.loc["default", "share_same_key"] == 1.0
    alts = res["alts"]
    assert (alts.groupby("variant").size().reindex(rows.index) == rows["n_alt"]).all()
    assert rows.loc["no_cap", "n_capped"] == 0
    assert (rows.loc["default", "before_cap"]
            == rows.loc["default", "n_alt"] + rows.loc["default", "n_capped"])
    with pytest.raises(ValueError):
        S.run_sensitivity(Net(), 0, BASE, links, stored, funnel_o, specs=SPECS,
                          log=lambda m: None, max_r3=1)


def test_r3_exact_implies_the_full_result_for_frequency_variants():
    """Where r3_exact holds on the MU = 1.0 set for a frequency variant, the
    variant gives the same alternatives on that set as on every candidate."""
    exact_seen = 0
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        full = V.number(synthetic_pair(rng))
        sub = V.number(full[S.r3_keep(full, 1.0)].drop(columns="cand"))
        for how in ("fixed", "span", "first"):
            spec = {"freq": how}
            seen = V.with_frequency(sub, BASE, how)
            if not S.r3_exact(seen, 1.0, BASE["MU"]):
                continue
            exact_seen += 1
            a_full, _ = V.decide_variant(full, BASE, spec)
            a_sub, _ = V.decide_variant(sub, BASE, spec)
            assert keys_of(full, a_full) == keys_of(sub, a_sub)
    assert exact_seen > 0


def test_cap_and_no_r4_mix_and_ranking():
    sa = pd.DataFrame({"origin": 1, "variant": "cross_class_no_r4",
                       "seq": ["C", "A", "A", "AA", "AR"]})
    m = S.no_r4_mix(sa, 1)
    assert m == {"n": 5, "car": 1, "A": 2, "AA": 1, "other": 1, "share": pytest.approx(0.6)}
    sens = pd.DataFrame({"origin": 1, "variant": ["default", "no_cap", "a", "b", "c"],
                         "share_same_key": [1.0, 1.0, 0.9, 0.5, 0.7]})
    assert [v for v, _ in S.rank_variants(sens, 1)] == ["b", "c", "a"]


def _sens_fixture(with_bands: bool):
    variants = ["default"] + [k for k in S.SENS_VARIANTS if k != "default"]
    share = {v: 0.9 for v in variants}
    share.update({"default": 1.0, "no_cap": 1.0, "cross_class_no_r4": 0.2,
                  "freq_first": 0.5, "freq_span": 0.6, "freq_fixed": 0.7})
    rows = []
    for o in (367, 148):
        for v in variants:
            rows.append({"origin": o, "variant": v, "pairs": 3, "n_alt": 30, "per_pair": 10.0,
                         "pairs_capped": 2, "n_capped": 4, "before_cap": 34,
                         "share_same_key": share[v], "share_found_member": share[v],
                         "pairs_not_exact": 1 if v in S.FREQ_VARIANTS else 0})
    sens = pd.DataFrame(rows)
    sa = pd.DataFrame({"origin": [367] * 4 + [148] * 4, "variant": "cross_class_no_r4",
                       "seq": ["C", "A", "AA", "AR"] * 2})
    runs = pd.DataFrame({"origin": [367, 148], "oded_removals": [2, 1], "eps_max": [70.0, 71.0]})
    if with_bands:
        runs = runs.assign(r2_removals=[20, 10], oded_eps_max=[30.0, 25.0],
                           r2_eps_max=[40.0, 35.0])
    funnel = pd.DataFrame({"o": [367, 148], "d": [1, 1], "n_r1": [25, 13], "n_r2": [5, 3]})
    x = {"sens": sens, "sens_alts": sa, "sens_runs": runs, "funnel": funnel,
         "params": {**BASE},
         "fuas": pd.DataFrame({"id": [367, 148], "name": ["Wien", "Katowice"]})}
    ctx = {"link_periods": {m: {"n": 10, "fixed_zero": z} for m, z in
                            (("A", 0), ("B", 9), ("R", 6))}}
    return x, ctx


@pytest.mark.parametrize("with_bands", [False, True])
def test_sensitivity_lessons_are_computed(with_bands):
    x, ctx = _sens_fixture(with_bands)
    text = " ".join(S.sensitivity_lessons(x, ctx))
    assert "freq_first and freq_span change the set most in both origins" in text
    assert "cross_class_no_r4 20.0%, freq_first 50.0%, freq_span 60.0%" in text
    assert "1 car, 1 direct flights, 1 with two flights (AA), 1 others" in text
    assert "9 of 10 coach links and 6 of 10 rail links" in text
    assert "Wien 2 of 20 R2 removals (10.00%)" in text
    assert ("bands up to 30.0 min" in text) == with_bands
    assert ("the bands of the removals themselves were not stored" in text) != with_bands
    # a gap between the stored removals and ps-funnel is reported
    x["funnel"].loc[0, "n_r2"] = 6
    assert ("NOT the same" in S.oded_lesson(x, ctx)) == with_bands


# --- Review items ----------------------------------------------------------------
def test_item_numbers_and_safe_item():
    assert S.item_no(S.REVIEW_ITEMS[0]) == S.N_NIGHT_ITEMS + 1 == 12
    assert S.item_no("rare") == 23
    def boom(x, ctx, out):
        raise KeyError("col")
    assert S.safe_item(boom, {}, {}, {}, "Label") == "Label. Not checked now: KeyError: 'col'."
    assert "known data hole".lower() in S.hole_note("Amsterdam", "Berlin").lower()
    assert f"item {S.item_no('eurostar')}" in S.hole_note("Paris", "London")
    assert S.hole_note("Nowhere", "Else") == ""


def test_overshoot_kinds():
    conn = pd.DataFrame({"fua": [5, 5], "hub": [30, 40], "acc_min": 1.0, "egr_min": [10.0, 10.0]})
    mode = np.array(["A", "R", "A", "B"])          # link id -> mode
    p = pd.DataFrame({"d": 5, "h_last": [40, 40, 40, 40],
                      "l1": [0, 0, 1, 1], "l2": [1, 2, 2, 3], "l3": -1,
                      "a1": [30, 30, 31, 32], "b2": [33, 30, 30, 34],
                      "a2": [-1, -1, -1, -1], "b3": [-1, -1, -1, -1]})
    # 0: lands at 30 (serves d), next link 1 is rail; 1: lands at 30, next is air;
    # 2: reaches 31 by rail, moves to 30 and boards a flight; 3: never at 30
    k = S.overshoot_kinds(p, conn, mode)
    assert k.tolist() == ["land_ground", "land_air", "board", ""]
    assert S.overshoots(p, conn).tolist() == [True, True, True, False]


def test_near_ties():
    a = pd.DataFrame({"o": 1, "d": 2, "seq": ["R", "R", "R", "B", "C"],
                      "T": [100.0, 110.0, 140.0, 104.0, 90.0],
                      "h_first": [1, 1, 3, 1, -1], "h_last": [2, 2, 2, 2, -1]})
    nt = S.near_ties(a, 15)
    assert nt == {"n": 1, "pairs": 1, "same_ends": 1, "within5": 0}


def test_distances_and_the_distance_table():
    wkb = [struct.pack("<BIdd", 1, 1, 2.3522, 48.8566), struct.pack("<BIdd", 1, 1, -0.1276, 51.5072),
           b"\x00"]
    c = S.fua_centres(pd.DataFrame({"id": [1, 2, 3], "centre": wkb}))
    assert c.loc[1, "lon"] == pytest.approx(2.3522) and np.isnan(c.loc[3, "lat"])
    assert S.pair_km([1], [2], c)[0] == pytest.approx(343.5, abs=2)
    assert S.band_labels([0, 150, 300, 99999]) == ["0-150", "150-300", "300 and more"]
    f = pd.DataFrame({"o": [1, 1], "d": [2, 3], "n_kept": [4, 2], "n_capped": [1, 0],
                      "n_classes": [3, 2]})
    al = pd.DataFrame({"o": 1, "d": [2, 2, 2, 3, 3], "seq": ["R", "AR", "C", "B", "B"],
                       "T": [100.0, 250.0, 90.0, 50.0, 60.0]})
    t = S.distance_table(f, al, [100.0, 400.0], [0, 150, 300, 99999]).set_index("km")
    assert t.loc["0-150", "pairs"] == 1 and t.loc["0-150", "capped"] == "100.0%"
    assert t.loc["0-150", "2 x least T or more"] == "50.0%"
    assert t.loc["0-150", "with a flight"] == "50.0%"
    assert t.loc["300 and more", "service alternatives"] == 2


def test_worker_estimate():
    w = S.worker_estimate([5083.0, 5083.0, 3601.7, 3500.0], 6)
    assert w["top"] == 5083.0 and w["second"] == 3601.7 and w["n_top"] == 2
    assert w["estimate"] == pytest.approx(5083 + 5 * 3601.7) and w["worst"] == 6 * 5083.0


def test_suspect_links_and_uses_links():
    links = pd.DataFrame({"link": [0, 1], "src": [0, 1], "trg": [1, 0], "ride": [121.0, 140.0]})
    names = np.array(["Paris Gare du Nord", "St-Pancras-International"], dtype=object)
    table = {0: ("Paris Gare du Nord", "St-Pancras-International", 121.0, "dwell"),
             1: ("St-Pancras-International", "Paris Gare du Nord", 121.0, "dwell"),
             7: ("x", "y", 1.0, "gone")}
    ok, stale = S.suspect_links(links, names, table)
    assert ok == {0: "dwell"} and stale == [1, 7]
    p = pd.DataFrame({"l1": [0, 1, -1], "l2": [-1, 0, -1], "l3": -1})
    assert S.uses_links(p, ok).tolist() == [True, True, False]


def test_direct_summary_and_text():
    hubs = pd.DataFrame({"hub": [1, 2, 3], "fua": [10, 20, 20]})
    links = pd.DataFrame({"src": [1, 1, 1], "trg": [2, 3, 2], "mode": ["R", "R", "B"],
                          "freq": [2.0, 3.0, 5.0], "active_dates": [100, 170, 50],
                          "first_day": [0, 10, 0], "last_day": [200, 300, 100]})
    sparse = links.iloc[:0].assign(active_dates=pd.Series(dtype=int))
    s = S.direct_summary(links, sparse, hubs, 10, 20, "R")
    assert s["links"] == 2 and s["per_day"] == 5.0 and s["dates_min"] == 100
    t = S.summary_text(s, "rail")
    assert t.startswith("2 rail links, 2 to 3 a day each (5 in all), on 100 to 170 dates")
    one = S.summary_text(S.direct_summary(links, sparse, hubs, 10, 20, "B"), "coach")
    assert one.startswith("1 coach link, 5 a day, on 50 dates")
    assert S.summary_text(S.direct_summary(links, sparse, hubs, 10, 20, "A"), "air") == "no air link"


def test_dwell_table_and_suspects():
    stations = pd.DataFrame({"stn": [1, 2], "name": ["Paris Nord", "London St Pancras"]})
    feeds = pd.DataFrame({"id": [5, 9], "name": ["man-EuroStar", "other"]})
    tt = pd.DataFrame({"jrn": [1, 1, 2, 2, 3, 3], "stn": [1, 2, 1, 2, 1, 2],
                       "arr": [1091, 1230, 1100, 1238, 1100, 1250],
                       "dep": [1109, 1230, 1100, 1238, 1100, 1250]})
    jr = pd.DataFrame({"jrn": [1, 2, 3], "feed": [5, 5, 9]})
    t = S.dwell_table(tt, jr, stations, feeds, "Paris Nord", "Pancras")
    sus, floor = S.dwell_suspects(t)
    assert floor == 138 and len(sus) == 1
    r = sus.iloc[0]
    assert (r["feed_name"], r["dwell"], r["ride"], r["ride_from_arr"]) == ("man-EuroStar", 18, 121, 139)


def test_direct_fate():
    best = pd.DataFrame({"seq": ["C", "R", "RR", "B"], "gtt": [200.0, 476.8, 449.0, 320.0],
                         "gtc": [200.0, 716.8, 499.0, 350.0]})
    alts = pd.DataFrame({"seq": ["C", "RR", "B", "BA"], "cls": ["C", "R", "B", "AB"],
                         "gtc": [200.0, 499.0, 350.0, 830.0]})
    txt = S.direct_fate("R", alts, best, BASE)
    assert "the cap dropped it" in txt and "BA (gtc 830)" in txt
    best2 = best.assign(gtt=[200.0, 700.0, 449.0, 320.0])
    assert "R3 cut it unless" in S.direct_fate("R", alts, best2, BASE)
    assert "CROSS_CLASS on" in S.direct_fate("R", alts, best, {**BASE, "CROSS_CLASS": True})


def test_remover_status_and_cache_floor():
    tr = pd.DataFrame({"o": 1, "d": 2, "cand": [10, 11, 12, 13],
                       "status": ["kept", "capped", "removed_by:10", "removed_by:11"],
                       "rule": ["", "R5", "R2", "R2"], "by": [-1, -1, 10, 11]})
    assert S.remover_status(tr).tolist() == ["kept", "capped"]
    attrs = {7: {"search": {"pass2": {"written": 1000, "triple_rows": 10}}}}
    held = 1000 * S.search.ID_BYTES + 10 * S.search.TV_BYTES
    assert S.cache_floor(attrs, 7, 1.0) == pytest.approx((2 ** 30 - held) / S.search.CACHE_BYTES)


def test_sens_code_hash_follows_the_variants(monkeypatch):
    h = S.sens_code_hash()
    assert len(h) == 16 and h == S.sens_code_hash()
    monkeypatch.setattr(S, "SENS_VARIANTS", {**S.SENS_VARIANTS, "x": {"params": {"MU": 0.3}}})
    assert S.sens_code_hash() != h
