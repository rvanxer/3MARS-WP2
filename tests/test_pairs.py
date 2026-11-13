"""Admissible link pairs and the transfer tables (pathset/pairs.py) on
synthetic networks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import dates as D  # noqa: E402
from pathset import pairs as PR  # noqa: E402
from pathset import schemas as S  # noqa: E402
from pathset import transfer as T  # noqa: E402
from pathset.io import P  # noqa: E402

WINDOWS = P["WINDOWS"]

# Hubs 0..7. City: 0 -> 10, 1 -> 11, 2 -> 12, 3 -> 11 (second hub of city 11),
# 4 -> 14, 5 -> 10 (second hub of the city of hub 0), 6 and 7 airports with
# cities 11 and 17.
CITY = np.array([10, 11, 12, 11, 14, 10, 11, 17], dtype=np.int64)
MOVES = pd.DataFrame({"hub_a": np.array([1, 1], dtype=np.int32),
                      "hub_b": np.array([3, 6], dtype=np.int32),
                      "minutes": np.array([10.0, 0.0], dtype=np.float32),
                      "kind": ["interhub", "colocated"]})


def link_table(rows):
    """ps-links columns link, src, trg, mode from (src, trg, mode)."""
    df = pd.DataFrame(rows, columns=["src", "trg", "mode"])
    df.insert(0, "link", np.arange(len(df), dtype=np.int32))
    return df


LINKS = link_table([
    (0, 1, "R"),   # 0 into hub 1
    (1, 2, "R"),   # 1 same hub
    (3, 4, "B"),   # 2 through the move 1 -> 3
    (1, 0, "R"),   # 3 straight back to src(link 0)
    (1, 5, "R"),   # 4 back into city 10 through another hub
    (6, 7, "A"),   # 5 flight from the airport co-located with hub 1
    (2, 6, "A"),   # 6 flight into airport 6
    (6, 4, "A"),   # 7 flight onward from airport 6
])


def pairs_of(links=LINKS, moves=MOVES, windows=WINDOWS):
    out, counts = PR.admissible_pairs(links, moves, CITY, windows)
    return out, counts, {(r.link_in, r.link_out): r for r in out.itertuples()}


def test_same_hub_and_move_pairs():
    out, _, by = pairs_of()
    assert by[(0, 1)].move == 0 and by[(0, 1)].kind == PR.SAME_HUB
    assert by[(0, 2)].move == 10 and by[(0, 2)].kind == "interhub"
    assert by[(0, 5)].move == 0 and by[(0, 5)].kind == "colocated"
    # no pair without a junction: link 1 ends at hub 2, link 2 starts at hub 3
    assert (1, 2) not in by
    assert out[["link_in", "link_out"]].apply(tuple, axis=1).is_unique


def test_both_exclusions():
    _, counts, by = pairs_of()
    assert (0, 3) not in by                       # trg(out) == src(in)
    assert (0, 4) not in by                       # back into city 10
    assert counts["trg(link_out) == src(link_in)"] >= 1
    assert counts["back into the city of src(link_in)"] >= 1
    # the city rule does not remove a pair whose ends are in other cities
    assert (6, 7) in by


def test_window_by_mode_pair():
    _, _, by = pairs_of()
    expect = {(0, 1): "PP", (0, 2): "PP", (0, 5): "PA", (6, 7): "AA",
              (6, 5): "AA"}
    for pair, key in expect.items():
        assert by[pair].wkey == key
        assert (by[pair].lo, by[pair].hi) == tuple(WINDOWS[key])
    # air -> ground: flight 6 lands at airport 6, a co-located move to hub 1,
    # then a coach (link 8) from hub 1 to hub 4
    links = link_table(list(LINKS[["src", "trg", "mode"]].itertuples(index=False))
                       + [(1, 4, "B")])
    _, _, by2 = pairs_of(links, pd.DataFrame({"hub_a": [6], "hub_b": [1],
                                              "minutes": [0.0], "kind": ["colocated"]}))
    assert (by2[(6, 8)].wkey, by2[(6, 8)].lo, by2[(6, 8)].hi) == ("AP", *WINDOWS["AP"])


def test_move_outside_window_is_dropped():
    tight = dict(WINDOWS, PP=[5, 12])             # 5 + 10 > 12
    _, counts, by = pairs_of(windows=tight)
    assert (0, 2) not in by and (0, 1) in by
    assert counts["lo + move > hi"] == 1
    _, counts, by = pairs_of(windows=dict(WINDOWS, PP=[5, 15]))   # 5 + 10 = 15
    assert (0, 2) in by and counts["lo + move > hi"] == 0


def test_typical_flag():
    _, _, by = pairs_of()
    assert by[(0, 5)].typical                     # rail -> air
    assert not by[(0, 1)].typical                 # rail -> rail
    assert not by[(0, 2)].typical                 # rail -> bus
    assert not by[(6, 7)].typical                 # air -> air


def test_dtypes_and_order():
    out, _, _ = pairs_of()
    assert out["link_in"].dtype == np.int32 and out["link_out"].dtype == np.int32
    assert out["move"].dtype == np.float32
    assert out["lo"].dtype == np.int32 and out["hi"].dtype == np.int32
    assert out["typical"].dtype == bool
    key = out["link_in"].astype(np.int64) * 1000 + out["link_out"]
    assert key.is_monotonic_increasing


def test_links_must_be_in_order():
    with pytest.raises(ValueError):
        PR.admissible_pairs(LINKS.iloc[::-1], MOVES, CITY, WINDOWS)


def test_pair_counts():
    out, _, _ = pairs_of()
    table = PR.pair_counts(out)
    assert table.loc["all", "all"] == len(out)
    assert table.loc["PP", PR.SAME_HUB] == ((out["wkey"] == "PP")
                                           & (out["kind"] == PR.SAME_HUB)).sum()


# --- End to end ---------------------------------------------------------------
def hm(h, m=0):
    return 60 * h + m


def test_end_to_end_with_move_and_air():
    """Hub 0 -> 1 by rail at 08:00 (60 min); from hub 1 a train at 09:20, from
    hub 3 (move 10 min) a coach at 09:30, from airport 6 (co-located with hub
    1) a flight at 11:00; all daily. Expected f_trans 20, 30 and 120, the
    first two date by date, the flight on typical days."""
    links = link_table([(0, 1, "R"), (1, 2, "R"), (3, 4, "B"), (6, 7, "A")])
    events = [(hm(8), 60, 0), (hm(9, 20), 60, 1), (hm(9, 30), 90, 2),
              (hm(11), 120, 3)]
    dep = np.array([e[0] for e in events], dtype=np.int16)
    ride = np.array([e[1] for e in events], dtype=np.int16)
    trip = np.array([e[2] for e in events], dtype=np.int32)
    masks = D.pack_bool(np.ones((len(events), D.N_DAYS), dtype=bool))
    ev0 = np.arange(4, dtype=np.int32)
    ev1 = ev0 + 1
    tz = np.zeros(4, dtype=np.int32)
    args = (dep, ride, masks, ev0, ev1, tz)
    inst, typ = T.build_instances(*args), T.build_typical(*args)
    cal = T.link_calendars(masks, ev0, ev1)

    pairs, _, _ = pairs_of(links)
    stats = T.transfer_table(inst, typ, cal, trip, pairs["link_in"].to_numpy(),
                             pairs["link_out"].to_numpy(), pairs["move"].to_numpy(),
                             pairs["lo"].to_numpy(), pairs["hi"].to_numpy(),
                             pairs["typical"].to_numpy())
    tr = PR.transfers_table(pairs, stats, T.basis_labels)
    assert list(tr.columns) == S.TRANSFERS
    assert [str(tr[c].dtype) for c in ("link_in", "link_out", "move", "n_days",
                                       "freq", "ft_mean", "ft_min")] == \
        ["int32", "int32", "float32", "int32", "float32", "float32", "float32"]
    got = {(r.link_in, r.link_out): r for r in tr.itertuples()}
    assert set(got) == {(0, 1), (0, 2), (0, 3)}
    assert (got[(0, 1)].basis, got[(0, 1)].ft_mean, got[(0, 1)].n_days,
            got[(0, 1)].freq) == ("D", 20, D.N_DAYS, 1)
    assert (got[(0, 2)].basis, got[(0, 2)].ft_mean, got[(0, 2)].move) == ("D", 30, 10)
    assert (got[(0, 3)].basis, got[(0, 3)].ft_mean, got[(0, 3)].n_days,
            got[(0, 3)].freq) == ("T", 120, 7, 1)
    assert PR.exists(tr, P["MIN_ACTIVE_DATES"]).all()

    # a pair without an opportunity is not kept: the coach leaves too early
    # for the move (09:05 < 09:00 + 5 + 10)
    dep2 = dep.copy()
    dep2[2] = hm(9, 5)
    inst2 = T.build_instances(dep2, ride, masks, ev0, ev1, tz)
    typ2 = T.build_typical(dep2, ride, masks, ev0, ev1, tz)
    stats2 = T.transfer_table(inst2, typ2, cal, trip, pairs["link_in"].to_numpy(),
                              pairs["link_out"].to_numpy(), pairs["move"].to_numpy(),
                              pairs["lo"].to_numpy(), pairs["hi"].to_numpy(),
                              pairs["typical"].to_numpy())
    tr2 = PR.transfers_table(pairs, stats2, T.basis_labels)
    assert set(zip(tr2["link_in"], tr2["link_out"])) == {(0, 1), (0, 3)}


# --- ps-ftrans ----------------------------------------------------------------
def hub_table(released):
    return pd.DataFrame({"hub": np.arange(len(released), dtype=np.int32),
                         "released_id": released})


def test_ftrans_pools_classes_and_shared_nodes():
    # hubs 0 and 1 share STN_A; hub 2 is STN_B; hubs 3 and 4 have no id
    hubs = hub_table(["STN_A", "STN_A", "STN_B", "", ""])
    links = link_table([(0, 2, "R"), (1, 2, "R"), (2, 0, "R"),
                        (3, 2, "B"), (4, 2, "B"), (2, 3, "R")])
    members = pd.DataFrame({"link": [0, 0, 1, 2, 3, 4, 5],
                            "agency": ["NS", "DB", "NS", "SNCF", "X", "Y", "Z"]})
    tr = pd.DataFrame({
        "link_in": np.array([0, 1, 3, 4], dtype=np.int32),
        "link_out": np.array([2, 2, 5, 5], dtype=np.int32),
        "move": np.zeros(4, dtype=np.float32),
        "basis": ["D", "T", "D", "D"],
        "n_days": np.array([100, 7, 10, 10], dtype=np.int32),
        "freq": np.array([2, 1, 1, 1], dtype=np.float32),
        "ft_mean": np.array([10, 40, 20, 30], dtype=np.float32),
        "ft_min": np.array([5, 30, 20, 30], dtype=np.float32)})
    ft = PR.ftrans_table(tr, links, hubs, members)
    assert list(ft.columns) == S.FTRANS
    pooled = ft[(ft["in_src"] == "STN_A") & (ft["out_trg"] == "STN_A")].iloc[0]
    assert pooled["n_rows"] == 2
    assert pooled["ft_mean"] == pytest.approx((200 * 10 + 7 * 40) / 207, rel=1e-6)
    assert pooled["ft_min"] == 5
    assert pooled["share_T"] == pytest.approx(0.5)
    assert pooled["in_ops"] == "DB;NS" and pooled["out_ops"] == "SNCF"
    # hubs 3 and 4 have no released id: their rows are left out
    assert len(ft) == 1 and (ft[S.FTRANS[:4]] != "").all(axis=None)
    assert ft["ft_mean"].dtype == np.float32 and ft["n_rows"].dtype == np.int32


def test_ftrans_key_unique_and_rows_without_released_id_left_out():
    """Review finding 1: a row with a hub without a released id at any of the
    four ends used to repeat the six-column key with ''. Such rows are left
    out of ps-ftrans (they stay in ps-transfers); the key is unique."""
    # hubs 0, 1 share STN_A; 2 is STN_B; 3 is STN_C; 4 and 5 have no id
    hubs = hub_table(["STN_A", "STN_A", "STN_B", "STN_C", "", ""])
    links = link_table([(0, 2, "R"), (1, 2, "R"),    # 0, 1 into STN_B
                        (2, 3, "R"),                 # 2 STN_B -> STN_C
                        (2, 4, "R"), (2, 5, "R"),    # 3, 4 to hubs without id
                        (4, 2, "B"), (5, 2, "B")])   # 5, 6 from hubs without id
    members = pd.DataFrame({"link": np.arange(7), "agency": list("abcdefg")})
    li = np.array([0, 1, 0, 1, 0, 5, 6], dtype=np.int32)
    lo = np.array([2, 2, 3, 3, 4, 2, 2], dtype=np.int32)
    n = len(li)
    tr = pd.DataFrame({
        "link_in": li, "link_out": lo, "move": np.zeros(n, dtype=np.float32),
        "basis": ["D"] * n, "n_days": np.full(n, 10, dtype=np.int32),
        "freq": np.ones(n, dtype=np.float32),
        "ft_mean": np.arange(10, 10 + n, dtype=np.float32),
        "ft_min": np.arange(10, 10 + n, dtype=np.float32)})
    keep = PR.released_rows(tr, links, hubs)
    assert keep.tolist() == [True, True, False, False, False, False, False]
    ft = PR.ftrans_table(tr, links, hubs, members)
    assert not ft.duplicated(S.FTRANS[:6]).any()
    assert (ft[S.FTRANS[:4]] != "").all(axis=None)
    assert int(ft["n_rows"].sum()) == int(keep.sum())
    assert ft[S.FTRANS[:4]].values.tolist() == [["STN_A", "STN_B", "STN_B", "STN_C"]]
    assert ft["in_ops"].iloc[0] == "a;b" and ft["out_ops"].iloc[0] == "c"


def test_ftrans_without_released_rows_is_empty():
    hubs = hub_table(["", "", "STN_B"])
    links = link_table([(0, 2, "R"), (2, 1, "R")])
    tr = pd.DataFrame({
        "link_in": np.array([0], dtype=np.int32), "link_out": np.array([1], dtype=np.int32),
        "move": np.zeros(1, dtype=np.float32), "basis": ["T"],
        "n_days": np.array([3], dtype=np.int32), "freq": np.ones(1, dtype=np.float32),
        "ft_mean": np.ones(1, dtype=np.float32), "ft_min": np.ones(1, dtype=np.float32)})
    ft = PR.ftrans_table(tr, links, hubs,
                         pd.DataFrame({"link": [0, 1], "agency": ["a", "b"]}))
    assert len(ft) == 0 and list(ft.columns) == S.FTRANS
