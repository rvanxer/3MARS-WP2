"""Review of pathset/rules.py: the whole procedure (R3, R1, R2, comparison
across classes, guarantee, October margin, cap) against an independent
one-candidate-at-a-time implementation of the task text, and the time-only
chain p, q, r of the 16 Oct note."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathset import rules  # noqa: E402
from test_rules import BASE, make, random_cands, run, status  # noqa: E402

KEPT = ("kept", "kept_guarantee")


def removes(p, q, eps_T, eps_M, rho, x):
    """R2 for one pair; money compared when the band is on and both known."""
    if p["n"] > q["n"] or p[x] > q[x] + eps_T:
        return False
    better = p["n"] < q["n"] or p[x] < q[x] - eps_T
    if eps_M is not None and not (math.isnan(p["M"]) or math.isnan(q["M"])):
        if p["M"] > q["M"] + eps_M:
            return False
        better = better or p["M"] < q["M"] - eps_M
    return better and (rho == 0 or p["freq"] >= rho * q["pooled"])


def stable(ids, dom):
    """Iterated elimination; returns the kept ids and, for every removed id,
    all kept ids that remove it."""
    alive, kept, gone = set(ids), [], set()
    while alive:
        free = [i for i in alive if not any(dom(j, i) for j in alive if j != i)]
        assert free, "cycle"
        kept += free
        alive -= set(free)
        hit = {q for q in alive if any(dom(p, q) for p in kept)}
        gone |= hit
        alive -= hit
    return set(kept), {q: {p for p in kept if dom(p, q)} for q in gone}


def brute(cands, P, order, guarantee):
    """The procedure as the task states it (with the implementer's two
    documented choices: R1_KEY = class keys on (class, n); the money band of
    R2 within a class needs M for all its alternatives)."""
    rows = {r["cand"]: dict(r, cls=rules.mode_class(r["seq"]))
            for r in cands.to_dict("records")}
    rank = {k: i for i, k in enumerate(sorted({r["key"] for r in rows.values()}))}
    classes = sorted({r["cls"] for r in rows.values()})
    mu, rho, eps_M = P["MU"], P["RHO_F"], P["EPS_M"]
    st, live = {}, set(rows)
    if order == "r3_first":
        for c, r in rows.items():
            same = [s for s in rows.values() if s["cls"] == r["cls"]]
            if (r["gtt"] > (1 + mu) * min(s["gtt"] for s in same)
                    and r["gtc"] > (1 + mu) * min(s["gtc"] for s in same)):
                st[c] = "cut_r3"
                live.discard(c)
    eps = {k: max(P["EPS_T_ABS"], P["EPS_T_REL"] * min(
        rows[c]["T"] for c in live if rows[c]["cls"] == k)) for k in classes}
    groups = {}
    for c in sorted(live, key=lambda c: (rows[c]["T"], rows[c]["gtt"],
                                         rank[rows[c]["key"]], c)):
        r = rows[c]
        key = r["seq"] if P["R1_KEY"] == "sequence" else (r["cls"], r["n"])
        for g in groups.setdefault(key, []):
            lead = rows[g[0]]
            ok = r["T"] - lead["T"] <= eps[r["cls"]]
            if eps_M is not None and not (math.isnan(r["M"]) or math.isnan(lead["M"])):
                ok = ok and abs(r["M"] - lead["M"]) <= eps_M
            if ok:
                g.append(c)
                break
        else:
            groups[key].append([c])
    alts = {}
    for g in itertools.chain.from_iterable(groups.values()):
        top = max(rows[m]["freq"] for m in g)
        rep = next(m for m in g if rho == 0 or rows[m]["freq"] >= rho * top)
        per_link = {}
        for m in g:
            fl = rows[m]["first_link"]
            per_link[fl] = max(per_link.get(fl, 0.0), rows[m]["freq"])
        alts[rep] = {**rows[rep], "pooled": sum(per_link.values())}
        st.update({m: f"variant_of:{rep}" for m in g if m != rep})
    x = rules.R2_TIMES[P["R2_TIME"]]
    okey = lambda a: (alts[a]["gtc"], rank[alts[a]["key"]], a)  # noqa: E731
    state, by = {}, {}
    for k in classes:
        ids = [a for a in alts if alts[a]["cls"] == k]
        band = None if any(math.isnan(alts[a]["M"]) for a in ids) else eps_M
        kept, rem = stable(ids, lambda p, q: removes(alts[p], alts[q], eps[k], band, rho, x))
        state.update({a: "kept" if a in kept else "removed_by" for a in ids})
        by.update(rem)
    if P["CROSS_CLASS"]:
        ids = [a for a in alts if state[a] == "kept"]
        zero = None if eps_M is None else 0
        kept, rem = stable(ids, lambda p, q: alts[p]["cls"] != alts[q]["cls"]
                           and removes(alts[p], alts[q], 0, zero, rho, x))
        state.update({a: "removed_by" for a in rem})
        by.update(rem)
        if guarantee:
            for k in classes:
                cid = [a for a in ids if alts[a]["cls"] == k]
                if cid and not kept & set(cid):
                    state[min(cid, key=okey)] = "kept_guarantee"
    if order == "october":
        for k in classes:
            cid = [a for a in alts if alts[a]["cls"] == k and state[a] in KEPT]
            if not cid:
                continue
            low = min(alts[a]["gtc"] for a in cid)
            state.update({a: "cut_r3" for a in cid if alts[a]["gtc"] > (1 + mu) * low})
    final = sorted((a for a in alts if state[a] in KEPT), key=okey)
    cap = max(P["K_CAP"], len(classes) + P["K_EXTRA"])
    while len(final) > cap:
        a = next((a for a in reversed(final)
                  if sum(alts[b]["cls"] == alts[a]["cls"] for b in final) > 1), None)
        if a is None:
            break
        final.remove(a)
        state[a] = "capped"
    st.update(state)
    return st, by, final


SETTINGS = [
    dict(CROSS_CLASS=True, K_CAP=4, K_EXTRA=1),
    dict(CROSS_CLASS=True, K_CAP=4, K_EXTRA=1, EPS_M=5),
    dict(CROSS_CLASS=True, RHO_F=0, EPS_M=5, K_CAP=3, K_EXTRA=0),
    dict(CROSS_CLASS=False, R1_KEY="class", RHO_F=1.0),
    dict(CROSS_CLASS=True, R2_TIME="weighted", EPS_M=10),
    dict(CROSS_CLASS=True, MU=0.1),
]


@pytest.mark.parametrize("setting", range(len(SETTINGS)))
def test_whole_procedure_against_brute_force(setting):
    P = {**BASE, **SETTINGS[setting]}
    rng = np.random.default_rng(500 + setting)
    for i in range(40):
        cands = random_cands(rng, ["none", "all", "some"][i % 3])
        for order, guarantee in itertools.product(rules.ORDERS, (True, False)):
            st, by, final = brute(cands, P, order, guarantee)
            alts, trace = rules.apply_rules(cands, P, order, guarantee)
            assert alts["alt"].tolist() == final
            got = status(trace)
            assert set(got) == set(st)
            for c, s in st.items():
                if s == "removed_by":
                    assert got[c].startswith("removed_by:")
                    assert int(got[c].split(":")[1]) in by[c]
                else:
                    assert got[c] == s


def test_time_only_chain_of_the_note():
    """p direct 120 min, q one transfer 110, r two transfers 100, eps 15:
    p removes q, q removes r, p does not remove r; the stable set is {p, r}."""
    rows = [{"cand": 1, "seq": "R", "T": 120}, {"cand": 2, "seq": "RR", "T": 110},
            {"cand": 3, "seq": "RRR", "T": 100}]
    alts, trace = run(rows, EPS_T_REL=0)
    assert set(alts["alt"]) == {1, 3}
    assert status(trace) == {1: "kept", 2: "removed_by:1", 3: "kept"}


def test_nullable_money_column():
    c = make([{"cand": 1, "seq": "R", "T": 250}, {"cand": 2, "seq": "R", "T": 290}])
    c["M"] = pd.array([50.0, None], dtype="Float64")
    alts, _ = rules.apply_rules(c, {**BASE, "EPS_M": 5})
    assert alts["alt"].tolist() == [1]
