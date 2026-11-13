"""Reviewer's cases for pathset/connect.py: edges the module tests leave out."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import connect as C  # noqa: E402
from pathset import schemas as S  # noqa: E402
from test_connect import (AIRPORTS, CENTRES, FITS, HUBS, NODES, PARAMS,  # noqa: E402
                          hub_table, km, released)


def test_one_direction_released_is_fitted_on_the_other_only():
    """Access released, egress missing: acc keeps the released minutes, egr
    is fitted, the row is flagged 'fitted'."""
    rel = released([("FUA_000", "STN_A", "Access", 6.0)])
    c = C.station_connectors(HUBS, rel, CENTRES, FITS).set_index("hub")
    assert c.loc[0, "acc_min"] == 6.0
    assert c.loc[0, "egr_min"] == pytest.approx(11 + km(4.0, 4.01))
    assert c.loc[0, "source"] == "fitted"


def test_hubs_sharing_a_released_node():
    """Two station hubs matched to one released node: both get its access
    minutes, and the move between them is fitted (no self link)."""
    hubs = hub_table([("S", 0, 4.010, 52.0, "", [1], "STN_A"),
                      ("S", 0, 4.016, 52.0, "", [2], "STN_A")])
    rel = released([("FUA_000", "STN_A", "Access", 6.0),
                    ("STN_A", "FUA_000", "Egress", 7.0)])
    c = C.station_connectors(hubs, rel, CENTRES, FITS)
    assert list(c["acc_min"]) == [6.0, 6.0] and list(c["source"]) == ["released"] * 2
    m = C.station_moves(hubs, rel, FITS["move"], PARAMS["MOVE_MAX_MIN"])
    assert set(m["kind"]) == {"fitted"} and len(m) == 2
    assert m["minutes"].iloc[0] == pytest.approx(5 + 2 * km(4.010, 4.016))


def test_no_moves_at_all_gives_an_empty_contract_table():
    """One station per FUA, no air2rail row, no released Interhub link."""
    hubs = hub_table([("S", 0, 4.01, 52.0, "", [1], "STN_A"),
                      ("A", -1, 4.5, 52.0, "AAA", [], "AIR_AAA")])
    rel = released([("FUA_000", "STN_A", "Access", 6.0)])
    a2r = pd.DataFrame({"iata": pd.Series([], dtype=str),
                        "stn": pd.Series([], dtype=np.int64),
                        "tier": pd.Series([], dtype=str)})
    m = C.build_moves(hubs, a2r, rel, FITS, PARAMS)
    assert list(m.columns) == S.MOVES and len(m) == 0


def test_air2rail_station_outside_every_hub_is_ignored():
    a2r = pd.DataFrame({"iata": ["AAA"], "stn": [999], "tier": ["airport"]})
    rel = released([])
    m = C.airport_moves(HUBS, a2r, rel, PARAMS["AIR2RAIL_COLOCATED_TIERS"],
                        PARAMS["AIR_ROAD_MOVE_MAX_MIN"])
    assert len(m) == 0


def test_released_airport_access_above_cap_is_not_replaced_by_the_fit():
    """FUA 1 -> AAA is released at 130 min; the fit would give about 54 min.
    The released value decides and the pair is dropped."""
    fit = FITS["A_acc"](km(5.0, 4.5))
    assert fit < 120
    rel = released([("FUA_001", "AIR_AAA", "Access", 130.0),
                    ("AIR_AAA", "FUA_001", "Egress", 131.0)])
    c = C.airport_connectors(HUBS, AIRPORTS, rel, CENTRES, FITS, 120)
    assert not ((c["hub"] == 4) & (c["fua"] == 1)).any()
    # the other catchment FUAs of AAA have no released link: fitted
    assert set(c.loc[c["hub"] == 4, "source"]) == {"fitted"}


def test_hub_city_ignores_station_connectors_of_other_fuas():
    """A station hub's city is its own FUA even if a connector row (invalid,
    but possible in hand-made input) links it to another FUA."""
    conn = pd.DataFrame({"fua": [5], "hub": [0], "acc_min": [1.0],
                         "egr_min": [1.0], "source": ["released"]})
    assert C.hub_city(HUBS, conn)[0] == 0


def test_build_is_deterministic_under_row_order():
    """Shuffling the input rows gives the same tables."""
    a2r = pd.DataFrame({"iata": ["AAA", "AAA"], "stn": [14, 12],
                        "tier": ["terminal", "airport"]})
    from test_connect import REL
    c1 = C.build_connectors(HUBS, AIRPORTS, REL, CENTRES, FITS, 120)
    m1 = C.build_moves(HUBS, a2r, REL, FITS, PARAMS)
    rel2 = C.Released(REL.access.iloc[::-1], REL.egress.iloc[::-1],
                      REL.interhub.iloc[::-1], NODES.set_index("node_id"))
    c2 = C.build_connectors(HUBS, AIRPORTS.iloc[::-1], rel2, CENTRES, FITS, 120)
    m2 = C.build_moves(HUBS, a2r.iloc[::-1], rel2, FITS, PARAMS)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(m1, m2)
