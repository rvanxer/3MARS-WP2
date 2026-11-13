"""Connectors and moves (pathset/connect.py) on synthetic inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import connect as C  # noqa: E402
from pathset import schemas as S  # noqa: E402
from pathset.network import gc_km  # noqa: E402

PARAMS = {"MOVE_MAX_MIN": 30, "AIR_ROAD_MOVE_MAX_MIN": 45,
          "AIR2RAIL_COLOCATED_TIERS": ["airport", "terminal"]}
CAP_AIR = 120
# Fits given by hand: minutes = a + b x km
FITS = {"S_acc": C.Fit(10.0, 1.0, 1, 0, 0, 3), "S_egr": C.Fit(11.0, 1.0, 1, 0, 0, 3),
        "A_acc": C.Fit(20.0, 1.0, 1, 0, 0, 3), "A_egr": C.Fit(21.0, 1.0, 1, 0, 0, 3),
        "move": C.Fit(5.0, 2.0, 1, 0, 0, 3)}


def hub_table(rows):
    """ps-hubs from (kind, fua, lon, lat, iata, stns, released_id)."""
    df = pd.DataFrame(rows, columns=["kind", "fua", "lon", "lat", "iata",
                                     "stns", "released_id"])
    df.insert(0, "hub", np.arange(len(df), dtype=np.int32))
    df["name"] = [f"h{i}" for i in range(len(df))]
    df["icc"], df["tz_std"], df["match_m"] = "NL", 1, np.nan
    df["stns"] = [np.asarray(s, dtype=np.int32) for s in df["stns"]]
    return df[S.HUBS]


# FUA 0 at (4.0, 52.0), FUA 1 at (5.0, 52.0), FUA 2 at (6.0, 52.0)
CENTRES = pd.DataFrame({"lon": [4.0, 5.0, 6.0], "lat": [52.0] * 3},
                       index=np.array([0, 1, 2]))
HUBS = hub_table([
    ("S", 0, 4.01, 52.0, "", [10], "STN_A"),      # 0 released
    ("S", 0, 4.05, 52.0, "", [11], "STN_B"),      # 1 released
    ("S", 0, 4.30, 52.0, "", [12], ""),           # 2 fitted
    ("S", 1, 5.02, 52.0, "", [13, 14], "STN_C"),  # 3 released node, no released access
    ("A", -1, 4.5, 52.0, "AAA", [], "AIR_AAA"),   # 4
    ("A", -1, 9.0, 52.0, "BBB", [], "AIR_BBB"),   # 5 far from everything
])
NODES = pd.DataFrame({"node_id": ["STN_A", "STN_B", "STN_C", "AIR_AAA", "AIR_BBB"],
                      "lon": [4.01, 4.05, 5.02, 4.5, 9.0], "lat": [52.0] * 5})


def released(rows):
    """Released graph from rows (src, trg, kind, minutes)."""
    links = pd.DataFrame(rows, columns=["src", "trg", "kind", "time"])
    return C.released_tables(NODES, links)


REL = released([
    ("FUA_000", "STN_A", "Access", 7.0), ("FUA_000", "STN_A", "Access", 6.0),
    ("STN_A", "FUA_000", "Egress", 8.0),
    ("FUA_000", "STN_B", "Access", 9.0), ("STN_B", "FUA_000", "Egress", 9.5),
    ("FUA_000", "AIR_AAA", "Access", 40.0), ("AIR_AAA", "FUA_000", "Egress", 41.0),
    ("FUA_001", "AIR_AAA", "Access", 130.0), ("AIR_AAA", "FUA_001", "Egress", 131.0),
    ("FUA_002", "AIR_AAA", "Access", 25.0), ("AIR_AAA", "FUA_002", "Egress", 26.0),
    ("STN_A", "STN_B", "Interhub", 12.0), ("STN_A", "STN_B", "Interhub", 11.0),
    ("STN_B", "STN_A", "Interhub", 31.0),
    ("AIR_AAA", "STN_C", "Interhub", 45.0), ("STN_C", "AIR_AAA", "Interhub", 46.0),
    ("AIR_AAA", "STN_B", "Interhub", 20.0), ("STN_B", "AIR_AAA", "Interhub", 20.0),
])
AIRPORTS = pd.DataFrame({"iata": ["AAA", "BBB"], "fua": [[0, 1, 2], [2]]})


def km(lon1, lon2, lat=52.0):
    return float(gc_km(lon1, lat, lon2, lat))


def test_fit_line_and_leave_one_out():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y = np.array([3.0, 5.5, 8.0, 17.0, 23.0])
    f = C.fit_line(x, y)
    b, a = np.polyfit(x, y, 1)
    assert f.a == pytest.approx(a) and f.b == pytest.approx(b) and f.n == 5
    loo = []
    for i in range(5):
        k = np.arange(5) != i
        bi, ai = np.polyfit(x[k], y[k], 1)
        loo.append(abs(y[i] - (ai + bi * x[i])))
    assert f.loo_mae == pytest.approx(np.mean(loo))
    exact = C.fit_line([0, 1, 2], [3, 5, 7])
    assert (exact.a, exact.b) == pytest.approx((3, 2)) and exact.r2 == pytest.approx(1)


def test_duplicates_give_minimum():
    acc = REL.access.set_index(["fua", "node"])["minutes"]
    assert acc[(0, "STN_A")] == 6.0
    ih = REL.interhub.set_index(["src", "trg"])["minutes"]
    assert ih[("STN_A", "STN_B")] == 11.0


def test_released_beats_fitted():
    c = C.station_connectors(HUBS, REL, CENTRES, FITS).set_index("hub")
    # hub 0: released 6 / 8 although the fit gives about 10.7
    assert (c.loc[0, "acc_min"], c.loc[0, "egr_min"], c.loc[0, "source"]) == (
        6.0, 8.0, "released")
    # hub 2: no released node -> fit at FUA centre -> hub
    d = km(4.0, 4.30)
    assert c.loc[2, "acc_min"] == pytest.approx(10 + d)
    assert c.loc[2, "egr_min"] == pytest.approx(11 + d)
    assert c.loc[2, "source"] == "fitted"
    # hub 3: released node but no released access link -> fitted
    assert c.loc[3, "source"] == "fitted"
    assert c.loc[3, "acc_min"] == pytest.approx(10 + km(5.0, 5.02))
    assert list(c["fua"]) == [0, 0, 0, 1]


def test_airport_cap_and_city():
    conn = C.build_connectors(HUBS, AIRPORTS, REL, CENTRES, FITS, CAP_AIR)
    assert list(conn.columns) == S.CONNECTORS
    a = conn[conn["hub"] == 4].set_index("fua")
    # FUA 1: released 130 > 120 -> dropped; FUA 0 and 2 released
    assert sorted(a.index) == [0, 2]
    assert (a.loc[2, "acc_min"], a.loc[2, "egr_min"]) == (25.0, 26.0)
    # BBB: fitted 20 + ~206 km > 120 -> no connector
    assert not (conn["hub"] == 5).any()
    city = C.hub_city(HUBS, conn)
    assert city.dtype == np.int64
    assert list(city) == [0, 0, 0, 1, 2, -6]   # AAA: FUA 2 (25 < 40); BBB: -1 - 5


def test_airport_fitted_within_cap_is_kept():
    near = AIRPORTS.assign(fua=[[0, 1, 2], [2]])
    hubs = HUBS.copy()
    hubs.loc[5, "lon"] = 6.5                    # ~34 km from FUA 2
    conn = C.build_connectors(hubs, near, REL, CENTRES, FITS, CAP_AIR)
    r = conn[conn["hub"] == 5].iloc[0]
    assert r["fua"] == 2 and r["source"] == "fitted"
    assert r["acc_min"] == pytest.approx(20 + km(6.0, 6.5), rel=1e-6)
    assert r["egr_min"] == pytest.approx(21 + km(6.0, 6.5), rel=1e-6)


def test_airport_city_ties_to_lower_fua():
    conn = pd.DataFrame({"fua": [2, 1, 0], "hub": [4, 4, 0],
                         "acc_min": [30.0, 30.0, 5.0], "egr_min": [1.0] * 3,
                         "source": "released"})
    city = C.hub_city(HUBS, conn)
    assert city[4] == 1 and city[5] == -6


def test_hub_city_skips_nan_access_and_matches_search():
    """An airport row with NaN acc_min does not make the city; the result is
    the array search.hub_cities builds (one rule, one convention)."""
    from pathset import search
    conn = pd.DataFrame({"fua": [0, 2, 1], "hub": [4, 4, 5],
                         "acc_min": [40.0, np.nan, np.nan],
                         "egr_min": [41.0, 26.0, 50.0], "source": "released"})
    city = C.hub_city(HUBS, conn)
    assert city[4] == 0 and city[5] == -6
    np.testing.assert_array_equal(city, search.hub_cities(HUBS, conn))
    full = C.build_connectors(HUBS, AIRPORTS, REL, CENTRES, FITS, CAP_AIR)
    np.testing.assert_array_equal(C.hub_city(HUBS, full),
                                  search.hub_cities(HUBS, full))


def test_airport_cap_per_direction():
    """ACCESS_MAX_AIR_MIN holds for each direction: the direction above the
    cap is NaN, the pair is kept while one direction is within it, and
    access_hubs / egress_hubs leave the NaN direction out."""
    rel = released([
        ("FUA_000", "AIR_AAA", "Access", 100.0), ("AIR_AAA", "FUA_000", "Egress", 125.0),
        ("FUA_001", "AIR_AAA", "Access", 121.0), ("AIR_AAA", "FUA_001", "Egress", 119.0),
        ("FUA_002", "AIR_AAA", "Access", 130.0), ("AIR_AAA", "FUA_002", "Egress", 131.0),
    ])
    conn = C.build_connectors(HUBS, AIRPORTS, rel, CENTRES, FITS, CAP_AIR)
    a = conn[conn["hub"] == 4].set_index("fua")
    assert sorted(a.index) == [0, 1]                       # FUA 2: both above
    assert a.loc[0, "acc_min"] == 100.0 and np.isnan(a.loc[0, "egr_min"])
    assert np.isnan(a.loc[1, "acc_min"]) and a.loc[1, "egr_min"] == 119.0
    assert (a["source"] == "released").all()
    hubs, _ = C.access_hubs(conn, 0)
    assert 4 in hubs
    hubs, _ = C.egress_hubs(conn, 0)
    assert 4 not in hubs
    hubs, _ = C.access_hubs(conn, 1)
    assert 4 not in hubs
    hubs, egr = C.egress_hubs(conn, 1)
    assert list(hubs) == [3, 4] and egr[1] == 119.0
    assert C.hub_city(HUBS, conn)[4] == 0                  # FUA 1 has no access


def test_fit_line_rejects_equal_distances():
    with pytest.raises(ValueError, match="distinct distances"):
        C.fit_line([5.0, 5.0, 5.0, 5.0], [10.0, 11.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="at least 3"):
        C.fit_line([1.0, 2.0], [3.0, 4.0])


def test_station_moves_and_cap():
    slow = C.Fit(5.0, 1.2, 1, 0, 0, 3)
    m = C.station_moves(HUBS, REL, slow, PARAMS["MOVE_MAX_MIN"])
    m = m.set_index(["hub_a", "hub_b"])
    assert m.loc[(0, 1), "minutes"] == 11.0 and m.loc[(0, 1), "kind"] == "interhub"
    assert (1, 0) not in m.index                # released 31 > 30, fit not used
    d02 = km(4.01, 4.30)                        # ~19.9 km -> 28.9 min
    assert m.loc[(0, 2), "minutes"] == pytest.approx(5 + 1.2 * d02)
    assert m.loc[(0, 2), "kind"] == "fitted"
    assert m.loc[(2, 0), "kind"] == "fitted"
    # nothing across FUAs
    assert not ({3} & set(m.index.get_level_values(0)))
    # with b = 2 the same pair needs 44.8 min > 30: no fitted move is left
    m = C.station_moves(HUBS, REL, FITS["move"], PARAMS["MOVE_MAX_MIN"])
    assert list(zip(m["hub_a"], m["hub_b"])) == [(0, 1)]


def test_airport_moves_colocated_and_road():
    a2r = pd.DataFrame({"iata": ["AAA", "AAA", "AAA"], "stn": [14, 10, 12],
                        "tier": ["terminal", "nearby", "airport"]})
    m = C.airport_moves(HUBS, a2r, REL, PARAMS["AIR2RAIL_COLOCATED_TIERS"],
                        PARAMS["AIR_ROAD_MOVE_MAX_MIN"]).set_index(["hub_a", "hub_b"])
    # stn 14 (member of hub 3, tier terminal) and stn 12 (hub 2, airport)
    for s in (3, 2):
        assert m.loc[(4, s), "kind"] == "colocated" and m.loc[(4, s), "minutes"] == 0
        assert m.loc[(s, 4), "kind"] == "colocated"
    # stn 10 is only 'nearby': hub 0 has no released road link -> no move
    assert (4, 0) not in m.index and (0, 4) not in m.index
    # hub 1: road both ways, released minutes
    assert m.loc[(4, 1), "kind"] == "road" and m.loc[(4, 1), "minutes"] == 20.0
    assert m.loc[(1, 4), "minutes"] == 20.0
    # no airport-to-airport moves
    assert not ((m.index.get_level_values(0) >= 4)
                & (m.index.get_level_values(1) >= 4)).any()


def test_airport_moves_may_leave_the_airport_city():
    """Airport-station moves reach station hubs of any catchment FUA of the
    airport, not only of hub_city(airport) (documented in airport_moves):
    AAA's city is FUA 2 (access 25 min), its road move goes to hub 1 in FUA 0
    and its co-located move to hub 3 in FUA 1."""
    a2r = pd.DataFrame({"iata": ["AAA"], "stn": [13], "tier": ["airport"]})
    m = C.build_moves(HUBS, a2r, REL, FITS, PARAMS).set_index(["hub_a", "hub_b"])
    city = C.hub_city(HUBS, C.build_connectors(HUBS, AIRPORTS, REL, CENTRES,
                                               FITS, CAP_AIR))
    assert city[4] == 2 and city[1] == 0 and city[3] == 1
    assert m.loc[(4, 1), "kind"] == "road" and m.loc[(4, 3), "kind"] == "colocated"


def test_road_cap_per_direction():
    a2r = pd.DataFrame({"iata": ["AAA"], "stn": [10], "tier": ["nearby"]})
    m = C.airport_moves(HUBS, a2r, REL, PARAMS["AIR2RAIL_COLOCATED_TIERS"],
                        PARAMS["AIR_ROAD_MOVE_MAX_MIN"]).set_index(["hub_a", "hub_b"])
    # AIR_AAA -> STN_C 45 (kept, at the cap); STN_C -> AIR_AAA 46 (dropped)
    assert m.loc[(4, 3), "minutes"] == 45.0 and m.loc[(4, 3), "kind"] == "road"
    assert (3, 4) not in m.index


def test_build_moves_contract():
    a2r = pd.DataFrame({"iata": ["AAA"], "stn": [14], "tier": ["airport"]})
    m = C.build_moves(HUBS, a2r, REL, FITS, PARAMS)
    assert list(m.columns) == S.MOVES
    assert m["hub_a"].dtype == np.int32 and m["minutes"].dtype == np.float32
    assert not m.duplicated(["hub_a", "hub_b"]).any()
    assert (m["hub_a"] != m["hub_b"]).all()
    # colocated replaces the road link of the same pair
    assert m.set_index(["hub_a", "hub_b"]).loc[(4, 3), "kind"] == "colocated"


def test_access_and_egress_hubs():
    conn = C.build_connectors(HUBS, AIRPORTS, REL, CENTRES, FITS, CAP_AIR)
    hubs, acc = C.access_hubs(conn, 0)
    assert list(hubs) == [0, 1, 2, 4] and acc[0] == 6.0 and acc[3] == 40.0
    hubs, egr = C.egress_hubs(conn, 2)
    assert list(hubs) == [4] and egr[0] == 26.0
    hubs, _ = C.access_hubs(conn, 99)
    assert len(hubs) == 0


def test_released_fits_on_lines():
    """Fits recover exact lines laid on the released links."""
    ids = [f"STN_{i}" for i in range(4)] + [f"AIR_{i}" for i in range(4)]
    lon = [4.1, 4.3, 4.6, 5.0] * 2
    nodes = pd.DataFrame({"node_id": ids, "lon": lon, "lat": [52.0] * 8})
    rows = []
    for n, x in zip(ids, lon):
        d = km(4.0, x)
        a = 10 if n.startswith("STN") else 20
        rows += [("FUA_000", n, "Access", a + 1.5 * d),
                 (n, "FUA_000", "Egress", a + 1 + 1.5 * d)]
    for i in range(3):
        rows.append((ids[i], ids[i + 1], "Interhub", 3 + 2 * km(lon[i], lon[i + 1])))
    rel = C.released_tables(nodes, pd.DataFrame(rows, columns=["src", "trg", "kind", "time"]))
    f = C.released_fits(rel, CENTRES)
    assert (f["S_acc"].a, f["S_acc"].b) == pytest.approx((10, 1.5))
    assert (f["S_egr"].a, f["S_egr"].b) == pytest.approx((11, 1.5))
    assert (f["A_acc"].a, f["A_egr"].a) == pytest.approx((20, 21))
    assert (f["move"].a, f["move"].b) == pytest.approx((3, 2))
    assert f["S_acc"].n == 4 and f["move"].n == 3


def test_fua_centres_from_wkb():
    fuas = pd.DataFrame({"id": [3, 7], "centre": shapely.to_wkb(
        shapely.points([1.0, 2.0], [50.0, 51.0]))})
    c = C.fua_centres(fuas)
    assert list(c.index) == [3, 7] and c.loc[7, "lat"] == 51.0
