"""Review of pathset/pairs.py: admissible pairs and ps-ftrans against
independent pandas references on random networks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import pairs as PR  # noqa: E402
from pathset.io import P  # noqa: E402

WINDOWS = P["WINDOWS"]


def random_network(rng, n_hubs=14, n_links=90, n_moves=25):
    """Hubs 0..n_hubs-1, the last four airports; links among stations (B, R)
    and among airports (A); moves station-station and airport-station; cities
    with repeats so that the city rule has work to do."""
    air = np.arange(n_hubs) >= n_hubs - 4
    city = rng.integers(0, 6, n_hubs).astype(np.int64)
    st, ap = np.flatnonzero(~air), np.flatnonzero(air)
    rows = []
    for _ in range(n_links):
        if rng.random() < 0.3:
            a, b = rng.choice(ap, 2, replace=False)
            rows.append((a, b, "A"))
        else:
            a, b = rng.choice(st, 2, replace=False)
            rows.append((a, b, str(rng.choice(["B", "R"]))))
    links = pd.DataFrame(rows, columns=["src", "trg", "mode"])
    links.insert(0, "link", np.arange(len(links), dtype=np.int32))
    mv = set()
    while len(mv) < n_moves:
        a = int(rng.integers(0, n_hubs))
        b = int(rng.integers(0, n_hubs))
        if a != b and not (air[a] and air[b]):
            mv.add((a, b))
    mv = sorted(mv)
    moves = pd.DataFrame({
        "hub_a": np.array([m[0] for m in mv], dtype=np.int32),
        "hub_b": np.array([m[1] for m in mv], dtype=np.int32),
        "minutes": rng.choice([0.0, 3.5, 12.0, 30.0, 45.0], len(mv)).astype(np.float32),
        "kind": rng.choice(["interhub", "fitted", "road", "colocated"], len(mv))})
    return links, moves, city


def reference_pairs(links, moves, city, windows):
    """Admissible pairs by merges: same hub, or a move; the two exclusions;
    the window by modes; lo + move <= hi; typical when exactly one is air."""
    lin = links.rename(columns={"link": "link_in", "src": "s_in", "trg": "t_in", "mode": "m_in"})
    lout = links.rename(columns={"link": "link_out", "src": "s_out", "trg": "t_out", "mode": "m_out"})
    same = lin.merge(lout, left_on="t_in", right_on="s_out").assign(move=0.0, kind="hub")
    mv = (lin.merge(moves.rename(columns={"minutes": "move"}), left_on="t_in", right_on="hub_a")
          .merge(lout, left_on="hub_b", right_on="s_out"))
    p = pd.concat([same, mv], ignore_index=True)
    p = p[(p["t_out"] != p["s_in"]) & (city[p["t_out"]] != city[p["s_in"]])]
    wkey = (np.where(p["m_in"] == "A", "A", "P").astype(object)
            + np.where(p["m_out"] == "A", "A", "P").astype(object))
    p = p.assign(wkey=wkey, lo=[windows[k][0] for k in wkey], hi=[windows[k][1] for k in wkey],
                 typical=(p["m_in"] == "A").to_numpy() != (p["m_out"] == "A").to_numpy())
    p = p[p["lo"] + p["move"].astype(float) <= p["hi"]]
    return p.sort_values(["link_in", "link_out"]).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(12))
def test_admissible_pairs_match_merge_reference(seed):
    rng = np.random.default_rng(seed)
    links, moves, city = random_network(rng)
    windows = dict(WINDOWS, PP=[5, 40]) if seed % 3 == 0 else WINDOWS
    got, counts = PR.admissible_pairs(links, moves, city, windows)
    want = reference_pairs(links, moves, city, windows)
    assert len(got) == len(want)
    for c in ("link_in", "link_out", "lo", "hi", "typical"):
        assert np.array_equal(got[c].to_numpy(), want[c].to_numpy()), c
    assert np.array_equal(got["move"].to_numpy(), want["move"].to_numpy(np.float32))
    assert list(np.asarray(got["wkey"]).astype(str)) == list(want["wkey"])
    assert list(np.asarray(got["kind"]).astype(str)) == list(want["kind"])
    assert (counts["cross product"] - counts["trg(link_out) == src(link_in)"]
            - counts["back into the city of src(link_in)"] - counts["lo + move > hi"]) == len(got)


def test_ftrans_matches_groupby_reference():
    rng = np.random.default_rng(3)
    links, moves, city = random_network(rng, n_links=60)
    n_h = len(city)
    ids = np.array([f"N{int(i) % 7}" if i % 5 else "" for i in range(n_h)], dtype=object)
    hubs = pd.DataFrame({"hub": np.arange(n_h, dtype=np.int32), "released_id": ids})
    pairs, _ = PR.admissible_pairs(links, moves, city, WINDOWS)
    n = len(pairs)
    tr = pd.DataFrame({
        "link_in": pairs["link_in"].to_numpy(), "link_out": pairs["link_out"].to_numpy(),
        "move": pairs["move"].to_numpy(),
        "basis": rng.choice(["D", "T"], n),
        "n_days": rng.integers(1, 50, n).astype(np.int32),
        "freq": rng.integers(1, 9, n).astype(np.float32),
        "ft_mean": rng.uniform(5, 200, n).astype(np.float32)})
    tr["ft_min"] = (tr["ft_mean"] - rng.uniform(0, 5, n)).astype(np.float32)
    members = pd.DataFrame({"link": np.repeat(links["link"].to_numpy(), 2),
                            "agency": rng.choice(["a", "b", "c", "d"], 2 * len(links))})
    got = PR.ftrans_table(tr, links, hubs, members)

    src, trg, mode = links["src"].to_numpy(), links["trg"].to_numpy(), links["mode"].to_numpy()
    li, lo = tr["link_in"].to_numpy(), tr["link_out"].to_numpy()
    g = tr.assign(a=ids[src[li]], b=ids[trg[li]], c=ids[src[lo]], d=ids[trg[lo]],
                  mi=mode[li], mo=mode[lo], w=tr["n_days"] * tr["freq"].astype(float))
    g = g[(g[["a", "b", "c", "d"]] != "").all(axis=1)]   # rows without a released end are left out
    assert 0 < len(g) < len(tr)
    g["wf"] = g["w"] * g["ft_mean"].astype(float)
    ops = members.groupby("link")["agency"].agg(set)
    rows = []
    for key, x in g.groupby(["a", "b", "c", "d", "mi", "mo"]):
        rows.append({
            "in_src": ids[src[x["link_in"].iloc[0]]], "in_trg": ids[trg[x["link_in"].iloc[0]]],
            "out_src": ids[src[x["link_out"].iloc[0]]], "out_trg": ids[trg[x["link_out"].iloc[0]]],
            "mode_in": key[4], "mode_out": key[5],
            "ft_mean": np.float32(x["wf"].sum() / x["w"].sum()),
            "ft_min": np.float32(x["ft_min"].min()), "n_rows": len(x),
            "share_T": np.float32((x["basis"] == "T").mean()),
            "in_ops": ";".join(sorted(set().union(*ops.reindex(x["link_in"].unique()).dropna()))),
            "out_ops": ";".join(sorted(set().union(*ops.reindex(x["link_out"].unique()).dropna())))})
    want = pd.DataFrame(rows)
    assert not got.duplicated(list(got.columns[:6])).any()
    cols = list(got.columns)
    a = got.sort_values(cols).reset_index(drop=True)
    b = want[cols].sort_values(cols).reset_index(drop=True)
    assert len(a) == len(b)
    for c in cols:
        if c in ("ft_mean",):
            assert np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-6)
        else:
            assert list(a[c]) == list(b[c]), c


# --- Check 10 of pathset-checks.py (review minor findings 2 and 3) -----------
def checks_module():
    """pathset-checks.py imported as a module (its name has a hyphen)."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "pathset-checks.py"
    spec = importlib.util.spec_from_file_location("pathset_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check10_hub_without_pairs_is_na_and_border_draw_is_measured():
    ck = checks_module()
    dd = {"n_days": np.array([10, 0, 30]), "freq": np.array([2.0, 0.0, 4.0])}
    tt = {"n_days": np.array([3, 0, 7]), "freq": np.array([2.0, 0.0, 5.0])}
    c = ck.comparison(dd, tt, 5)
    text, ok = ck.verdict(np.zeros(3, dtype=bool), c)
    assert ok is None and "n/a" in text
    _, ok = ck.verdict(np.ones(3, dtype=bool), c)      # agree 100%, median error 12.5%
    assert ok is False
    _, ok = ck.verdict(np.array([True, True, False]), c)
    assert ok is True
    pool, measured = np.arange(100), np.arange(0, 100, 3)
    pick = ck.draw_border(pool, measured, 20, 7)
    assert len(pick) == 20 and np.isin(pick, measured).all()
    assert (np.diff(pick) > 0).all()
    assert np.array_equal(ck.draw_border(pool, measured, 20, 7), pick)
    assert len(ck.draw_border(pool, np.array([4, 5, 500]), 20, 7)) == 2


def test_check10_meaning_is_computed_from_the_result():
    ck = checks_module()
    d = np.array([True] * 10 + [False] * 10)
    t = np.array([True] * 8 + [False] * 2 + [True] + [False] * 9)
    s = ck.air_ground_meaning(d, t, 0.05)
    assert "misses 20.0%" in s and "invents 11.1%" in s
    assert "a lower bound" in s and "median 5.0%" in s
    assert "an upper bound" in ck.air_ground_meaning(t, d, 0.0)   # 9 found date by date, 10 typical
