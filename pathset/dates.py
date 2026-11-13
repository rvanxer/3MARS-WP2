"""Timeline and calendar masks.

Day index 0 is ``SERVICE_START``; the timeline has ``N_DAYS`` days. A calendar
mask is a row of ``N_WORDS`` unsigned 64-bit words in which day ``d`` is bit
``d % 64`` of word ``d // 64``. Weekdays are numbered Monday = 0 ... Sunday = 6.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from numba import njit

from . import io

DAY0: dt.date = io.P["SERVICE_START"]
DAY1: dt.date = io.P["SERVICE_END"]
N_DAYS: int = (DAY1 - DAY0).days + 1
N_WORDS: int = (N_DAYS + 63) // 64
WEEKDAY0: int = DAY0.weekday()

_FULL = np.uint64(0xFFFFFFFFFFFFFFFF)


# --- Dates -------------------------------------------------------------------
def day_index(date: dt.date) -> int:
    return (date - DAY0).days


def index_date(i: int) -> dt.date:
    return DAY0 + dt.timedelta(days=int(i))


def date_columns() -> list[str]:
    """Column names of ic-datesets (yyyymmdd) in timeline order."""
    return [index_date(i).strftime("%Y%m%d") for i in range(N_DAYS)]


def weekdays() -> np.ndarray:
    """Weekday (Monday = 0) of every day of the timeline."""
    return ((np.arange(N_DAYS) + WEEKDAY0) % 7).astype(np.int8)


def _last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def eu_summer_days() -> np.ndarray:
    """True for days in EU summer time (from the last Sunday of March up to,
    not including, the last Sunday of October)."""
    out = np.zeros(N_DAYS, dtype=bool)
    for year in range(DAY0.year, DAY1.year + 1):
        a = max(day_index(_last_sunday(year, 3)), 0)
        b = min(day_index(_last_sunday(year, 10)), N_DAYS)
        if b > a:
            out[a:b] = True
    return out


# --- Packing -----------------------------------------------------------------
def pack_bool(mat: np.ndarray) -> np.ndarray:
    """Boolean matrix (n, N_DAYS) -> masks (n, N_WORDS) of uint64."""
    mat = np.asarray(mat, dtype=bool)
    if mat.ndim != 2 or mat.shape[1] != N_DAYS:
        raise ValueError(f"expected (n, {N_DAYS}), got {mat.shape}")
    padded = np.zeros((mat.shape[0], N_WORDS * 64), dtype=bool)
    padded[:, :N_DAYS] = mat
    packed = np.packbits(padded, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64, copy=False)


def unpack(masks: np.ndarray) -> np.ndarray:
    """Masks (n, N_WORDS) -> boolean matrix (n, N_DAYS)."""
    masks = np.ascontiguousarray(masks, dtype="<u8")
    bits = np.unpackbits(masks.view(np.uint8), axis=1, bitorder="little")
    return bits[:, :N_DAYS].astype(bool)


# --- Kernels -----------------------------------------------------------------
@njit(cache=True)
def popcount64(x):
    x = np.uint64(x)
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@njit(cache=True)
def popcount_row(mask):
    c = 0
    for w in range(mask.shape[0]):
        c += popcount64(mask[w])
    return c


@njit(cache=True)
def popcount_rows(masks):
    out = np.empty(masks.shape[0], dtype=np.int32)
    for i in range(masks.shape[0]):
        out[i] = popcount_row(masks[i])
    return out


@njit(cache=True)
def get_bit(mask, d):
    return (mask[d >> 6] >> np.uint64(d & 63)) & np.uint64(1) != np.uint64(0)


@njit(cache=True)
def and_any(a, b):
    """True when two masks share at least one day."""
    for w in range(a.shape[0]):
        if a[w] & b[w]:
            return True
    return False


@njit(cache=True)
def and_count(a, b):
    """Number of days two masks share."""
    c = 0
    for w in range(a.shape[0]):
        c += popcount64(a[w] & b[w])
    return c


@njit(cache=True)
def or_into(dst, src):
    for w in range(dst.shape[0]):
        dst[w] |= src[w]


@njit(cache=True)
def shift_row(src, k, out, n_days):
    """Shift a calendar by k days (day d becomes day d + k); days that leave
    the timeline are dropped."""
    n = src.shape[0]
    full = np.uint64(0xFFFFFFFFFFFFFFFF)
    for w in range(n):
        out[w] = np.uint64(0)
    if k == 0:
        for w in range(n):
            out[w] = src[w]
    elif k > 0:
        q = k // 64
        r = np.uint64(k % 64)
        for w in range(n - 1, -1, -1):
            sw = w - q
            if sw < 0:
                continue
            if r == np.uint64(0):
                out[w] = src[sw]
            else:
                v = src[sw] << r
                if sw - 1 >= 0:
                    v |= src[sw - 1] >> (np.uint64(64) - r)
                out[w] = v
    else:
        kk = -k
        q = kk // 64
        r = np.uint64(kk % 64)
        for w in range(n):
            sw = w + q
            if sw >= n:
                continue
            if r == np.uint64(0):
                out[w] = src[sw]
            else:
                v = src[sw] >> r
                if sw + 1 < n:
                    v |= src[sw + 1] << (np.uint64(64) - r)
                out[w] = v
    extra = n * 64 - n_days
    if extra > 0:
        out[n - 1] &= full >> np.uint64(extra)


@njit(cache=True)
def _shift_rows(masks, k, n_days):
    out = np.zeros_like(masks)
    for i in range(masks.shape[0]):
        shift_row(masks[i], k[i], out[i], n_days)
    return out


def shift_masks(masks: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Shift every calendar ``masks[i]`` by ``k[i]`` days."""
    masks = np.ascontiguousarray(masks, dtype=np.uint64)
    k = np.ascontiguousarray(k, dtype=np.int64)
    return _shift_rows(masks, k, N_DAYS)


@njit(cache=True)
def days_of(mask, n_days):
    """Day indices set in a mask, ascending."""
    out = np.empty(popcount_row(mask), dtype=np.int32)
    j = 0
    for d in range(n_days):
        if (mask[d >> 6] >> np.uint64(d & 63)) & np.uint64(1):
            out[j] = d
            j += 1
    return out
