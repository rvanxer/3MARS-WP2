"""Calendar masks: packing, shifting, counting."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathset import dates as D  # noqa: E402


def test_timeline():
    assert D.DAY0 == dt.date(2023, 1, 1)
    assert D.N_DAYS == 1461
    assert D.N_WORDS == 23
    # 1 January 2023 was a Sunday
    assert D.weekdays()[0] == 6
    assert D.weekdays()[1] == 0
    assert D.date_columns()[0] == "20230101"
    assert D.date_columns()[-1] == "20261231"
    assert D.day_index(dt.date(2026, 9, 8)) == D.date_columns().index("20260908")


def test_summer_time():
    s = D.eu_summer_days()
    assert not s[D.day_index(dt.date(2026, 3, 28))]
    assert s[D.day_index(dt.date(2026, 3, 29))]      # last Sunday of March
    assert s[D.day_index(dt.date(2026, 10, 24))]
    assert not s[D.day_index(dt.date(2026, 10, 25))]  # last Sunday of October
    assert not s[D.day_index(dt.date(2023, 1, 15))]
    assert s[D.day_index(dt.date(2024, 7, 1))]


def test_pack_roundtrip():
    rng = np.random.default_rng(1)
    mat = rng.random((50, D.N_DAYS)) < 0.3
    masks = D.pack_bool(mat)
    assert masks.shape == (50, D.N_WORDS) and masks.dtype == np.uint64
    assert (D.unpack(masks) == mat).all()
    assert (D.popcount_rows(masks) == mat.sum(axis=1)).all()
    for d in (0, 63, 64, 700, D.N_DAYS - 1):
        assert D.get_bit(masks[3], d) == mat[3, d]
    assert (D.days_of(masks[7], D.N_DAYS) == np.flatnonzero(mat[7])).all()


def test_and_or():
    rng = np.random.default_rng(2)
    a = rng.random((1, D.N_DAYS)) < 0.05
    b = rng.random((1, D.N_DAYS)) < 0.05
    ma, mb = D.pack_bool(a)[0], D.pack_bool(b)[0]
    assert D.and_any(ma, mb) == bool((a & b).any())
    assert D.and_count(ma, mb) == int((a & b).sum())
    mc = ma.copy()
    D.or_into(mc, mb)
    assert (D.unpack(mc[None, :])[0] == (a | b)[0]).all()


def test_shift():
    rng = np.random.default_rng(3)
    mat = rng.random((40, D.N_DAYS)) < 0.2
    ks = np.array([0, 1, -1, 2, 63, 64, 65, -64, -65, 130, -200, 1460, -1460,
                   2000, -2000] + list(rng.integers(-300, 300, 25)))
    shifted = D.unpack(D.shift_masks(D.pack_bool(mat), ks))
    for i, k in enumerate(ks):
        want = np.zeros(D.N_DAYS, dtype=bool)
        if k >= 0:
            if k < D.N_DAYS:
                want[k:] = mat[i, : D.N_DAYS - k]
        else:
            if -k < D.N_DAYS:
                want[: D.N_DAYS + k] = mat[i, -k:]
        assert (shifted[i] == want).all(), k
