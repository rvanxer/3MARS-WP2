"""Viewer and stage B (pathset/view.py) and the R1 key "pattern" of
pathset/rules.py, on synthetic tables."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import cost as C  # noqa: E402
from pathset import io, rules  # noqa: E402
from pathset import view as V  # noqa: E402

BASE = {**io.P, "EPS_T_ABS": 15, "EPS_T_REL": 0.05, "EPS_M": None, "MU": 0.5,
        "K_CAP": 8, "K_EXTRA": 3, "RHO_F": 0.5, "CROSS_CLASS": False,
        "R1_KEY": "sequence", "R2_TIME": "clock", "R4_GUARANTEE": True}

FUAS = pd.DataFrame({"id": [7, 162, 55, 105, 106, 378],
                     "name": ["Amsterdam", "København", "Bruxelles / Brussel",
                              "Frankfurt am Main", "Frankfurt (Oder)",
                              "s' Gravenhage"]})


# --- resolve_fua -------------------------------------------------------------
def test_resolve_fua_names_and_ids():
    assert V.resolve_fua("Amsterdam", FUAS) == 7
    assert V.resolve_fua("  amsterdam ", FUAS) == 7
    assert V.resolve_fua("7", FUAS) == 7
    assert V.resolve_fua("København", FUAS) == 162
    assert V.resolve_fua("KOBENHAVN", FUAS) == 162
    assert V.resolve_fua("Brussel", FUAS) == 55
    assert V.resolve_fua("bruxelles", FUAS) == 55
    assert V.resolve_fua("Frankfurt am Main", FUAS) == 105
    assert V.resolve_fua("'s-Gravenhage", FUAS) == 378
    assert V.resolve_fua("oder", FUAS) == 106


def test_resolve_fua_refuses_ambiguous_and_unknown():
    with pytest.raises(ValueError, match="ambiguous.*Frankfurt am Main.*Frankfurt \\(Oder\\)"):
        V.resolve_fua("frankfurt", FUAS)
    with pytest.raises(ValueError, match="no FUA named"):
        V.resolve_fua("Atlantis", FUAS)
    with pytest.raises(ValueError, match="no FUA with id"):
        V.resolve_fua("999", FUAS)


# --- Synthetic candidates ----------------------------------------------------
def cand_table(rows):
    """PATH-like candidates from short dicts (seq, T required); gtt = T,
    gtc = gtt + hidden wait at freq, t_ivt = T unless given."""
    par = C.param_array(BASE)
    out = []
    for i, r in enumerate(rows):
        car = r["seq"] == "C"
        freq = r.get("freq", math.inf if car else 1.0)
        gtt = r.get("gtt", r["T"])
        hwt = 0.0 if car else float(C.hidden_wait(freq, par))
        out.append({
            "seq": r["seq"], "n": 0 if car else len(r["seq"]) - 1,
            "h_first": r.get("h_first", -1 if car else 1),
            "h_last": r.get("h_last", -1 if car else 2),
            "l1": -1 if car else r.get("l1", 10 + i), "l2": r.get("l2", -1),
            "l3": -1, "cities": r.get("cities", ""), "key": r.get("key", f"k{i:03d}"),
            "t_ivt": r.get("t_ivt", r["T"]), "T": r["T"], "gtt": gtt, "hwt": hwt,
            "gtc": gtt + hwt, "freq": freq,
            "f_fixed": r.get("f_fixed", freq), "f_span": r.get("f_span", freq),
            "f_first": r.get("f_first", freq)})
    return V.number(pd.DataFrame(out))


# --- deck_top5 ---------------------------------------------------------------
def test_deck_top5_per_hub_pair_and_sequence():
    rows = [{"seq": "R", "T": 300, "t_ivt": t} for t in (70, 10, 60, 20, 50, 30, 40)]
    rows += [{"seq": "R", "T": 300, "t_ivt": 5, "h_first": 3},
             {"seq": "R", "T": 300, "t_ivt": 6, "h_first": 3},
             {"seq": "RR", "T": 320, "t_ivt": 1, "l2": 99},
             {"seq": "C", "T": 400}]
    c = cand_table(rows)
    sel = V.deck_top5(c, BASE)
    g = sel.groupby(["h_first", "h_last", "seq"]).size()
    assert g[(1, 2, "R")] == 5 and g[(3, 2, "R")] == 2
    assert g[(1, 2, "RR")] == 1 and g[(-1, -1, "C")] == 1
    first = sel[(sel["h_first"] == 1) & (sel["seq"] == "R")]
    assert sorted(first["t_ivt"]) == [10, 20, 30, 40, 50]
    assert first.sort_values("deck_rank")["t_ivt"].tolist() == [10, 20, 30, 40, 50]
    assert len(V.deck_top5(c, {**BASE, "DECK_TOP_K": 2})) == 2 + 2 + 1 + 1


# --- R1 key "pattern" --------------------------------------------------------
def rule_input(rows):
    return pd.DataFrame([{"cand": i, "seq": r["seq"], "n": len(r["seq"]) - 1,
                          "T": r["T"], "gtt": r["T"], "gtc": r["T"], "freq": 1.0,
                          "first_link": i, "key": f"{i:03d}", "cities": r["cities"]}
                         for i, r in enumerate(rows)])


def test_r1_pattern_keeps_transfer_cities_apart():
    """Two one-transfer rail paths through different cities, 5 minutes
    apart: one alternative under 'sequence' (16 Oct R1 merges near-ties
    across transfer cities), two under 'pattern'."""
    cands = rule_input([{"seq": "RR", "T": 300, "cities": "7>28>99"},
                        {"seq": "RR", "T": 305, "cities": "7>55>99"}])
    a_seq, t_seq = rules.apply_rules(cands, {**BASE, "R1_KEY": "sequence"})
    a_pat, t_pat = rules.apply_rules(cands, {**BASE, "R1_KEY": "pattern"})
    assert len(a_seq) == 1 and t_seq["status"].tolist() == ["kept", "variant_of:0"]
    assert len(a_pat) == 2 and set(t_pat["status"]) == {"kept"}


def test_r1_pattern_still_merges_within_one_pattern_and_checks_key():
    cands = rule_input([{"seq": "RR", "T": 300, "cities": "7>28>99"},
                        {"seq": "RR", "T": 305, "cities": "7>28>99"},
                        {"seq": "BR", "T": 302, "cities": "7>28>99"}])
    alts, trace = rules.apply_rules(cands, {**BASE, "R1_KEY": "pattern"})
    assert trace["status"].tolist() == ["kept", "variant_of:0", "kept"]
    with pytest.raises(ValueError):
        rules.check_params({**BASE, "R1_KEY": "corridor"})


# --- decide_variant ----------------------------------------------------------
def variant_table():
    return cand_table([
        {"seq": "C", "T": 600},
        {"seq": "R", "T": 450, "freq": 10, "f_first": 2, "f_fixed": 0},
        {"seq": "RR", "T": 380, "freq": 20, "cities": "7>28>99", "l2": 50,
         "f_first": 20, "f_fixed": 20},
        {"seq": "RR", "T": 385, "freq": 20, "cities": "7>55>99", "l2": 51,
         "f_first": 20, "f_fixed": 20},
        {"seq": "R", "T": 700, "freq": 1},                    # cut by R3 at MU 0.5
    ])


def test_decide_variant_default_and_pattern():
    c = variant_table()
    alts, trace = V.decide_variant(c, BASE, "default")
    st = dict(zip(trace["cand"], trace["status"]))
    assert len(alts) == 3                                     # car, R 450, RR group
    assert st[4] == "cut_r3"
    alts_p, _ = V.decide_variant(c, BASE, "r1_pattern")
    assert len(alts_p) == 4
    cmp_ = V.compare(alts, alts_p)
    assert cmp_["n_same"] == 3 and cmp_["n_base_found"] == 3 and cmp_["n_other_found"] == 4
    steps = dict(V.funnel(trace, alts))
    assert steps == {"R0": 5, "R3": 4, "R1": 3, "R2": 3, "cap": 3}


def test_decide_variant_frequency_recomputes_hwt_and_gtc():
    c = variant_table()
    par = C.param_array(BASE)
    alts, _ = V.decide_variant(c, BASE, "freq_first")
    r = alts.set_index("alt").loc[1]
    assert r["freq"] == pytest.approx(2)
    assert r["gtc"] == pytest.approx(450 + float(C.hidden_wait(2.0, par)), rel=1e-6)
    alts_f, _ = V.decide_variant(c, BASE, "freq_fixed")
    r = alts_f.set_index("alt").loc[1]
    assert r["freq"] == 0 and r["gtc"] == pytest.approx(450 + BASE["HWT_MAX"])
    car = alts_f.set_index("alt").loc[0]
    assert math.isinf(car["freq"]) and car["gtc"] == pytest.approx(600)
    with pytest.raises(ValueError, match="f_span"):
        V.decide_variant(c.drop(columns=["f_span"]), BASE, "freq_span")


def test_decide_variant_params_order_and_deck():
    c = variant_table()
    alts, trace = V.decide_variant(c, BASE, "mu_1.0")
    assert "cut_r3" not in set(trace["status"])               # gtt 700 <= 2 x 380
    alts_o, trace_o = V.decide_variant(c, BASE, "october")
    # October order: R2 runs before the margin, so the slow direct is removed
    # by the faster direct instead of being cut by R3
    assert dict(zip(trace_o["cand"], trace_o["status"]))[4] == "removed_by:1"
    assert dict(V.funnel(trace_o, alts_o, "october"))["R0"] == 5
    alts_d, trace_d = V.decide_variant(c, BASE, "deck_top5")
    assert len(alts_d) == len(c) and set(trace_d["status"]) == {"selected"}
    with pytest.raises(ValueError, match="unknown variant"):
        V.decide_variant(c, BASE, "nothing")


def test_route_names_rides_and_moves():
    names = np.array(["A", "B", "C", "D"], dtype=object)
    row = {"seq": "RR", "b1": 0, "a1": 1, "b2": 2, "a2": 3, "b3": -1, "a3": -1}
    assert V.route(row, names) == "A > B ~ C > D"
    row = {"seq": "RR", "b1": 0, "a1": 1, "b2": 1, "a2": 3, "b3": -1, "a3": -1}
    assert V.route(row, names) == "A > B > D"
    assert V.route({"seq": "C"}, names) == "car"


# --- link_freqs, link_rides on a synthetic ps-links --------------------------
LINKS = pd.DataFrame({"link": [2, 0, 1, 3],                 # any row order
                      "src": [2, 0, 1, 1], "trg": [3, 1, 2, 3],
                      "freq": [8.0, 20.0, 5.0, 3.0],
                      "freq_fixed": [6.0, 0.0, 4.0, 2.0],
                      "freq_span": [7.0, 18.0, 4.5, 2.5],
                      "ride": [30.0, 10.0, 20.0, 45.0]})


def link_paths():
    return pd.DataFrame({"seq": ["C", "R", "RR", "RRR"],
                         "l1": [-1, 3, 0, 0], "l2": [-1, -1, 3, 1],
                         "l3": [-1, -1, -1, 2]})


def test_link_freqs_least_over_used_links_first_link_and_car():
    out = V.link_freqs(link_paths(), LINKS)
    assert np.isinf(out.loc[0, ["f_fixed", "f_span", "f_first"]].astype(float)).all()
    assert out.loc[1, "f_fixed"] == 2 and out.loc[1, "f_span"] == 2.5
    assert out.loc[1, "f_first"] == 3
    assert out.loc[2, "f_fixed"] == 0 and out.loc[2, "f_span"] == 2.5   # min(18, 2.5)
    assert out.loc[2, "f_first"] == 20
    assert out.loc[3, "f_fixed"] == 0 and out.loc[3, "f_span"] == 4.5   # min(18, 4.5, 7)
    assert out.loc[3, "f_first"] == 20
    with pytest.raises(ValueError, match="0 .. L-1"):
        V.link_freqs(link_paths(), LINKS[LINKS["link"] != 1])


def test_link_rides_and_route_hubs():
    c = V.link_rides(link_paths(), LINKS)
    assert np.isnan(c.loc[0, V.RIDE_COLS].astype(float)).all()
    assert c.loc[3, V.RIDE_COLS].tolist() == [10.0, 20.0, 30.0]
    assert V.rides(c.loc[3]) == "10+20+30" and V.rides(c.loc[0]) == ""
    lk = LINKS.sort_values("link")
    h = V.link_hubs(c, lk["src"].to_numpy(), lk["trg"].to_numpy())
    names = np.array(["A", "B", "C", "D"], dtype=object)
    assert V.route(h.loc[3], names) == "A > B > C > D"
    assert V.route(h.loc[2], names) == "A > B > D"


# --- removed_table, funnel in the October order ---------------------------------
def removed_case():
    """Six alternatives: one kept, two capped and two removed by R2 (gtc in
    the opposite order of T), one cut by the October margin."""
    c = pd.DataFrame({"cand": np.arange(6), "seq": ["R"] * 6,
                      "T": [100.0, 150, 140, 300, 200, 120],
                      "gtt": [100.0, 150, 140, 300, 200, 120],
                      "gtc": [110.0, 500, 400, 450, 300, 900], "freq": 1.0,
                      "b1": 0, "a1": 1, "b2": -1, "a2": -1, "b3": -1, "a3": -1})
    trace = pd.DataFrame({
        "cand": np.arange(6), "alt": np.arange(6),
        "status": ["kept", "capped", "capped", "removed_by:0", "removed_by:0", "cut_r3"],
        "rule": ["", "R5", "R5", "R2", "R2", "R3"], "by": [-1, -1, -1, 0, 0, -1],
        "detail": ["", "rank 3", "rank 2", "n 0 <= 0", "n 0 <= 0", "gtc 900 > 165"]})
    return c, trace


def test_removed_table_by_rule_then_gtc_and_head_per_rule():
    c, trace = removed_case()
    names = np.array(["A", "B"], dtype=object)
    rem = V.removed_table(c, trace, names)
    assert rem["cand"].tolist() == [2, 1, 4, 3, 5]          # R5 by gtc, R2 by gtc, R3
    assert rem["rule"].tolist() == ["R5", "R5", "R2", "R2", "R3"]
    assert rem.loc[2, "by_route"] == "A > B"
    head = V.removed_head(rem, 1)
    assert head["cand"].tolist() == [2, 4, 5]
    assert V.rule_counts(rem) == "R5 2, R2 2, R3 1"
    empty = V.removed_table(c, trace.assign(status="kept"), names)
    assert len(empty) == 0 and "detail" in empty.columns


def test_funnel_october_order():
    c = variant_table()
    alts, trace = V.decide_variant(c, BASE, "october")
    # R1: car, R 450, R 700, RR {380, 385}; R2: R 450 removes R 700; the
    # margin on gtc keeps the rest; no cap
    assert V.funnel(trace, alts, "october") == [("R0", 5), ("R1", 4), ("R2", 3),
                                                ("R3", 3), ("cap", 3)]


def test_alternatives_table_components_and_hwt():
    c = variant_table()
    for k in V.COMPONENTS:
        c[k] = 0.0
    c["t_ivt"] = c["gtt"]
    c = V.link_hubs(c, np.arange(200), np.arange(200) + 1)
    alts, _ = V.decide_variant(c, BASE, "default")
    tab = V.alternatives_table(c, alts, np.array([f"h{i}" for i in range(202)], dtype=object))
    assert tab["rank"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(tab["hwt"], tab["gtc"] - tab["gtt"])
    np.testing.assert_allclose(tab[V.COMPONENTS].sum(axis=1), tab["gtt"])
    assert tab.loc[tab["seq"] == "C", "route"].tolist() == ["car"]
    assert {"rides", "key", "freq_pooled"} <= set(tab.columns)


# --- Parameters: PASSED_SHARE, sensitivity, deck K ----------------------------
def test_choose_passed_share_order_of_sources():
    recs = [{"step": "enumerate", "time": "t1", "info": {"mu": 0.5}},
            {"step": "enumerate", "time": "t2", "info": {"passed_share": 0.5}},
            {"step": "stage-b", "time": "t3", "info": {"passed_share": 1.0}}]
    no_key = {k: v for k, v in BASE.items() if k != "PASSED_SHARE"}
    assert V.choose_passed_share(0.25, no_key, recs) == (0.25, "command line")
    assert V.choose_passed_share(None, {**no_key, "PASSED_SHARE": 0.7}, recs) == (
        0.7, "pathset-params.yml")
    share, src = V.choose_passed_share(None, no_key, recs)
    assert share == 0.5 and "t2" in src
    assert V.choose_passed_share(None, no_key, recs[:1])[0] == 1.0
    with pytest.raises(ValueError):
        V.choose_passed_share(1.5, no_key, recs)


def test_variants_read_sensitivity_and_deck_k_from_params():
    no_keys = {k: v for k, v in BASE.items() if not k.startswith("STAGE_B_")}
    names = list(V.variants(no_keys))
    assert names[:len(V.RULE_VARIANTS)] == list(V.RULE_VARIANTS)
    assert "gate_pooled" in names
    assert {"eps_10", "eps_20", "mu_0.25", "mu_1.0", "rho_0.33", "rho_1.0"} <= set(names)
    p = {**no_keys, "STAGE_B_MU": [0.3], "STAGE_B_RHO_F": [0.4, 0.8]}
    v = V.variants(p)
    assert "mu_0.3" in v and "mu_0.25" not in v and v["rho_0.8"] == {
        "params": {"RHO_F": 0.8}}
    assert set(V.variant_notes(p)) == set(v)
    assert "STAGE_B_MU" not in V.missing_parameters(p)
    c = variant_table()
    alts, _ = V.decide_variant(c, p, "rho_0.8")
    assert len(alts) >= 1


# --- Real FUA names (data-upstream) ------------------------------------------
def test_resolve_fua_real_names():
    try:
        fuas = V.load_fuas()
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"data-upstream fuas not readable: {exc}")
    assert V.resolve_fua("Amsterdam", fuas) == 7
    assert V.resolve_fua("berlin", fuas) == 28
    assert V.resolve_fua("helsinki", fuas) == 133
    assert V.resolve_fua("TALLINN", fuas) == 325
    assert V.resolve_fua("Paris", fuas) == 254
