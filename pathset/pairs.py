"""Admissible ordered pairs of service links, and the tables ps-transfers and
ps-ftrans built on their statistics.

Plan step 2, "Transfer table"; plan, Definitions, "Transfer"; contract of
ps-transfers and ps-ftrans in ``pathset/schemas.py``. Every function is pure
and vectorised (no Python loop over pairs); reading, counting and writing are
done by ``pathset-transfers.py``.

Admissible pair (link_in, link_out):
  junction  trg(link_in) == src(link_out) with move 0 (one hub), or a ps-moves
            row from trg(link_in) to src(link_out) with move = its minutes;
  left out  trg(link_out) == src(link_in) (straight back), and
            city(trg(link_out)) == city(src(link_in)) with city =
            ``connect.hub_city`` (back into the city one came from). Both are
            removed by the search's R0 in any position of a path (c_first !=
            c_last for a two-link path; "no city of a transfer is c_first or
            c_last" for either transfer of a three-link path), so leaving them
            out here loses nothing;
  window    [lo, hi] = WINDOWS[schemas.window_key(mode_in, mode_out)] (deck
            slide 234); a pair with lo + move > hi has no feasible transfer
            and is left out (the move must fit inside the window: plan,
            Definitions, "Transfer");
  typical   exactly one of the two links is air: such a pair is always counted
            on typical days (plan, Definitions, "Typical days": "always for
            air <-> ground").

ps-ftrans holds only the ps-transfers rows whose four hubs all have a
released node id: it is keyed on the released 3MG graph (plan step 2, "Written
also at 3MG's link key with the released node ids"), and a row with a hub that
has no released id cannot be joined to it. Such rows stay in ps-transfers.
Every ps-ftrans key (in_src, in_trg, out_src, out_trg, mode_in, mode_out) is
therefore unique and no id is ''.

Weight of a row in the pooled f_trans (contract, ps-ftrans.ft_mean): freq x
n_days / days counted, the opportunities per counted day. Days counted are
the two links' common active dates on basis 'D' (the dates on which
transfer.pair_stats counts) and the 7 days of the typical week on basis 'T'
(``days_counted``). A row on basis 'D' counted over 1,000 common dates and
one on 'T' over one week thus weigh alike when they offer as many
opportunities a day; with freq x n_days alone the 'D' row weighed up to
1,000 / 7 times more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as S

SAME_HUB = "hub"             # move kind of a transfer inside one hub
TYPICAL_DAYS = 7             # days of the typical week (basis 'T')
ROWS_PER_CHUNK = 1 << 20     # rows of ps-transfers per step of days_counted

_TRANSFER_DTYPES = {"link_in": "int32", "link_out": "int32", "move": "float32",
                    "n_days": "int32", "freq": "float32", "ft_mean": "float32",
                    "ft_min": "float32"}


# --- Junctions and the cross product -----------------------------------------
def junctions(n_hubs: int, moves: pd.DataFrame) -> pd.DataFrame:
    """Places where one link can hand over to the next: (h, h, 0, 'hub') for
    every hub, then every ps-moves row (hub_a, hub_b, minutes, kind)."""
    hub = np.arange(n_hubs, dtype=np.int64)
    same = pd.DataFrame({"hub_a": hub, "hub_b": hub, "move": 0.0,
                         "kind": SAME_HUB})
    mv = pd.DataFrame({"hub_a": moves["hub_a"].to_numpy(np.int64),
                       "hub_b": moves["hub_b"].to_numpy(np.int64),
                       "move": moves["minutes"].to_numpy(np.float64),
                       "kind": moves["kind"].to_numpy(dtype=object)})
    return pd.concat([same, mv], ignore_index=True)


def _csr(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows grouped by key (stable, so ascending within a key) and pointers."""
    order = np.argsort(keys, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n), out=ptr[1:])
    return order, ptr


def cross(src: np.ndarray, trg: np.ndarray, hub_a: np.ndarray,
          hub_b: np.ndarray, n_hubs: int):
    """Every (link arriving at hub_a[j], link leaving hub_b[j]) of every
    junction j: arrays link_in, link_out, junction (int64)."""
    src = np.asarray(src, dtype=np.int64)
    trg = np.asarray(trg, dtype=np.int64)
    in_order, in_ptr = _csr(trg, n_hubs)
    out_order, out_ptr = _csr(src, n_hubs)
    n_in = in_ptr[hub_a + 1] - in_ptr[hub_a]
    n_out = out_ptr[hub_b + 1] - out_ptr[hub_b]
    cnt = n_in * n_out
    j = np.repeat(np.arange(len(hub_a), dtype=np.int64), cnt)
    k = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    no = n_out[j]
    link_in = in_order[in_ptr[hub_a[j]] + k // no]
    link_out = out_order[out_ptr[hub_b[j]] + k % no]
    return link_in, link_out, j


# --- Windows -----------------------------------------------------------------
def window_bounds(air_in: np.ndarray, air_out: np.ndarray, windows: dict):
    """lo, hi (int32) and the window key of every pair from whether each link
    is air: WINDOWS[schemas.window_key(mode_in, mode_out)] (deck slide 234)."""
    code = 2 * np.asarray(air_in, dtype=np.int64) + np.asarray(air_out, dtype=np.int64)
    keys = [S.window_key("A" if a else "R", "A" if b else "R")
            for a in (False, True) for b in (False, True)]
    lo = np.array([windows[k][0] for k in keys], dtype=np.int32)[code]
    hi = np.array([windows[k][1] for k in keys], dtype=np.int32)[code]
    return lo, hi, pd.Categorical.from_codes(code, keys)


# --- Admissible pairs --------------------------------------------------------
def admissible_pairs(links: pd.DataFrame, moves: pd.DataFrame, city: np.ndarray,
                     windows: dict) -> tuple[pd.DataFrame, dict]:
    """Admissible ordered pairs of service links (module docstring), sorted by
    (link_in, link_out); one row per pair.

    links: link (0..L-1 in order), src, trg, mode; moves: ps-moves; city:
    ``connect.hub_city``; windows: WINDOWS. Columns: link_in, link_out int32,
    move float32, lo, hi int32, typical bool, wkey (window key) and kind (move
    kind, 'hub' inside one hub) categorical. Also returns the number of
    cross-product pairs and of the pairs left out by each rule, in the order
    the rules are applied."""
    if not (links["link"].to_numpy() == np.arange(len(links))).all():
        raise ValueError("links must be ps-links in link order, 0 .. L-1")
    src = links["src"].to_numpy(np.int64)
    trg = links["trg"].to_numpy(np.int64)
    air = links["mode"].to_numpy() == "A"
    n_hubs = len(city)
    jn = junctions(n_hubs, moves)
    li, lo_, j = cross(src, trg, jn["hub_a"].to_numpy(), jn["hub_b"].to_numpy(),
                       n_hubs)
    counts = {"cross product": int(len(li))}

    back = trg[lo_] == src[li]
    counts["trg(link_out) == src(link_in)"] = int(back.sum())
    same_city = ~back & (city[trg[lo_]] == city[src[li]])
    counts["back into the city of src(link_in)"] = int(same_city.sum())
    keep = ~(back | same_city)
    li, lo_, j = li[keep], lo_[keep], j[keep]
    del back, same_city, keep

    move = jn["move"].to_numpy(np.float64)[j]
    lo, hi, wkey = window_bounds(air[li], air[lo_], windows)
    fits = lo + move <= hi
    counts["lo + move > hi"] = int((~fits).sum())

    kinds, kind_code = np.unique(jn["kind"].to_numpy(dtype=str), return_inverse=True)
    order = np.lexsort((lo_[fits], li[fits]))
    out = pd.DataFrame({
        "link_in": li[fits][order].astype(np.int32),
        "link_out": lo_[fits][order].astype(np.int32),
        "move": move[fits][order].astype(np.float32),
        "lo": lo[fits][order], "hi": hi[fits][order],
        "typical": (air[li] != air[lo_])[fits][order],
        "wkey": wkey[fits][order],
        "kind": pd.Categorical.from_codes(kind_code[j[fits][order]], kinds)})
    return out, counts


def pair_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    """Number of pairs by window key and move kind (rows and columns), with
    totals."""
    t = pairs.groupby(["wkey", "kind"], observed=False).size().unstack(fill_value=0)
    t.loc["all"] = t.sum(axis=0)
    t["all"] = t.sum(axis=1)
    return t


def memory_mb(df: pd.DataFrame) -> float:
    """Memory of a table in MB (deep: strings counted)."""
    return float(df.memory_usage(deep=True).sum()) / 2 ** 20


# --- ps-transfers ------------------------------------------------------------
def transfers_table(pairs: pd.DataFrame, stats: dict, labels) -> pd.DataFrame:
    """ps-transfers (contract): the pairs with at least one opportunity
    (n_days >= 1), in contract order and dtypes, keeping the order of
    ``pairs`` (sorted by link_in, link_out). ``stats`` is the output of
    ``transfer.transfer_table`` for ``pairs``; ``labels`` turns its basis
    codes into 'D' / 'T' (``transfer.basis_labels``)."""
    has = stats["n_days"] >= 1
    out = pd.DataFrame({
        "link_in": pairs["link_in"].to_numpy()[has],
        "link_out": pairs["link_out"].to_numpy()[has],
        "move": pairs["move"].to_numpy()[has],
        "basis": labels(stats["basis"][has]),
        "n_days": stats["n_days"][has], "freq": stats["freq"][has],
        "ft_mean": stats["ft_mean"][has], "ft_min": stats["ft_min"][has]})
    return out[S.TRANSFERS].astype(_TRANSFER_DTYPES)


def exists(transfers: pd.DataFrame, min_active: int) -> np.ndarray:
    """The consumer's rule (contract, ps-transfers): a two-link chain exists
    when n_days >= min_active (MIN_ACTIVE_DATES) on basis 'D' and n_days >= 1
    on basis 'T'."""
    need = np.where(transfers["basis"].to_numpy() == "D", min_active, 1)
    return transfers["n_days"].to_numpy() >= need


# --- ps-ftrans ---------------------------------------------------------------
def days_counted(transfers: pd.DataFrame, cal: np.ndarray) -> np.ndarray:
    """Days over which each ps-transfers row was counted (int32): on basis
    'D' the active dates the two links share, popcount(cal[link_in] &
    cal[link_out]) with ``cal`` = transfer.link_calendars (the dates of
    transfer.pair_stats); on basis 'T' TYPICAL_DAYS. Rows are taken
    ROWS_PER_CHUNK at a time to bound the memory of the calendar rows."""
    li = transfers["link_in"].to_numpy(np.int64)
    lo = transfers["link_out"].to_numpy(np.int64)
    cal = np.ascontiguousarray(cal, dtype=np.uint64)
    out = np.full(len(transfers), TYPICAL_DAYS, dtype=np.int32)
    rows = np.flatnonzero(transfers["basis"].to_numpy() == "D")
    for s in range(0, len(rows), ROWS_PER_CHUNK):
        r = rows[s:s + ROWS_PER_CHUNK]
        out[r] = np.bitwise_count(cal[li[r]] & cal[lo[r]]).sum(axis=1, dtype=np.int64)
    return out


def released_codes(hubs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per hub (indexed by hub 0..H-1): an integer code that is equal for
    hubs with the same released node id and -1 for a hub without one; and the
    released id ('' when none)."""
    hubs = hubs.sort_values("hub")
    if not (hubs["hub"].to_numpy() == np.arange(len(hubs))).all():
        raise ValueError("hubs must be ps-hubs with hub ids 0 .. H-1")
    rid = hubs["released_id"].fillna("").to_numpy(dtype=str)
    _, code = np.unique(rid, return_inverse=True)
    code = code.astype(np.int64)
    code[rid == ""] = -1
    return code, rid


def released_rows(transfers: pd.DataFrame, links: pd.DataFrame,
                  hubs: pd.DataFrame) -> np.ndarray:
    """Rows of ps-transfers whose four hubs (src and trg of link_in and of
    link_out) all have a released node id: the rows ps-ftrans pools."""
    code, _ = released_codes(hubs)
    src = links["src"].to_numpy(np.int64)
    trg = links["trg"].to_numpy(np.int64)
    li = transfers["link_in"].to_numpy(np.int64)
    lo = transfers["link_out"].to_numpy(np.int64)
    return ((code[src[li]] >= 0) & (code[trg[li]] >= 0)
            & (code[src[lo]] >= 0) & (code[trg[lo]] >= 0))


def _joined(group: np.ndarray, link: np.ndarray, members: pd.DataFrame,
            n_groups: int) -> np.ndarray:
    """Per group, the sorted distinct agencies of its links, ';'-joined (''
    for a group whose links have no member row).

    Integer keys throughout: the distinct (group, link), each expanded to the
    agency codes of its link (codes in name order), the distinct (group,
    agency); the strings are made by one join and one split."""
    names, code = np.unique(members["agency"].to_numpy(dtype=str), return_inverse=True)
    if any(";" in s or "\n" in s for s in names):
        raise ValueError("an agency name contains ';' or a line break")
    n_a = len(names)
    la = np.unique(members["link"].to_numpy(np.int64) * n_a + code)
    m_link, m_agency = la // n_a, la % n_a
    n_links = int(max(m_link.max(initial=-1), np.max(link, initial=-1))) + 1
    ptr = np.zeros(n_links + 1, dtype=np.int64)
    np.cumsum(np.bincount(m_link, minlength=n_links), out=ptr[1:])
    gl = np.unique(np.asarray(group, dtype=np.int64) * n_links + link)
    g, l = gl // n_links, gl % n_links
    cnt = ptr[l + 1] - ptr[l]
    rows = np.repeat(ptr[l], cnt) + (np.arange(int(cnt.sum()))
                                     - np.repeat(np.cumsum(cnt) - cnt, cnt))
    ga = np.unique(np.repeat(g, cnt) * n_a + m_agency[rows])
    g, a = ga // n_a, ga % n_a
    last = np.diff(g, append=g[-1:] + 1) != 0       # last agency of each group
    text = "".join(s for pair in zip(names[a].tolist(),
                                     np.where(last, "\n", ";").tolist())
                   for s in pair)
    out = np.full(n_groups, "", dtype=object)
    out[g[last]] = text.split("\n")[:-1]
    return out


def ftrans_table(transfers: pd.DataFrame, links: pd.DataFrame,
                 hubs: pd.DataFrame, members: pd.DataFrame,
                 days: np.ndarray | None = None) -> pd.DataFrame:
    """ps-ftrans (contract): ps-transfers at the key of the released 3MG graph.

    Only the rows whose four hubs all have a released node id are used
    (``released_rows``; module docstring). They are mapped to (in_src,
    in_trg, out_src, out_trg) = the released node ids of src(link_in),
    trg(link_in), src(link_out), trg(link_out) and the two modes, and pooled
    per that key over riding-time classes and over hubs that share a released
    node: ft_mean = mean of the rows' ft_mean weighted by freq x n_days /
    days counted (``days``, one per row of ``transfers``, from
    ``days_counted``; module docstring), ft_min = least ft_min, n_rows,
    share_T = share of rows on basis 'T', in_ops / out_ops = sorted distinct
    agencies (ps-link-members) of the pooled in- and out-links, ';'-joined.
    The key is unique; sorted by it.

    Without ``days`` the weight is freq x n_days (the weight before 13 Nov,
    which tests/test_transfers_review.py checks); pathset-transfers.py always
    passes the days."""
    if not (links["link"].to_numpy() == np.arange(len(links))).all():
        raise ValueError("links must be ps-links in link order, 0 .. L-1")
    if days is not None and len(days) != len(transfers):
        raise ValueError("days must have one entry per row of transfers")
    if days is not None and (np.asarray(days) <= 0).any():
        raise ValueError("days counted must be positive")
    code, rid = released_codes(hubs)
    keep = released_rows(transfers, links, hubs)
    src = links["src"].to_numpy(np.int64)
    trg = links["trg"].to_numpy(np.int64)
    mode_codes, mode = np.unique(links["mode"].to_numpy(dtype=str), return_inverse=True)
    li = transfers["link_in"].to_numpy(np.int64)[keep]
    lo = transfers["link_out"].to_numpy(np.int64)[keep]
    nh, nm = int(code.max()) + 1, len(mode_codes)
    key = (((code[src[li]] * nh + code[trg[li]]) * nh + code[src[lo]]) * nh
           + code[trg[lo]]) * nm * nm + mode[li] * nm + mode[lo]
    uniq, first, g = np.unique(key, return_index=True, return_inverse=True)
    n = len(uniq)

    def col(name):
        return transfers[name].to_numpy(np.float64)[keep]

    w = col("n_days") * col("freq")
    if days is not None:
        w = w / np.asarray(days, dtype=np.float64)[keep]
    ft = col("ft_mean")
    ft_min = np.full(n, np.inf)
    np.minimum.at(ft_min, g, col("ft_min"))
    n_rows = np.bincount(g, minlength=n)
    is_t = (transfers["basis"].to_numpy()[keep] == "T").astype(np.float64)
    fi, fo = li[first], lo[first]
    out = pd.DataFrame({
        "in_src": rid[src[fi]], "in_trg": rid[trg[fi]],
        "out_src": rid[src[fo]], "out_trg": rid[trg[fo]],
        "mode_in": mode_codes[mode[fi]], "mode_out": mode_codes[mode[fo]],
        "ft_mean": (np.bincount(g, w * ft, minlength=n)
                    / np.bincount(g, w, minlength=n)).astype(np.float32),
        "ft_min": ft_min.astype(np.float32),
        "n_rows": n_rows.astype(np.int32),
        "share_T": (np.bincount(g, is_t, minlength=n) / n_rows).astype(np.float32),
        "in_ops": _joined(g, li, members, n),
        "out_ops": _joined(g, lo, members, n)})
    out = out.astype({c: "str" for c in ("in_src", "in_trg", "out_src", "out_trg",
                                         "mode_in", "mode_out", "in_ops", "out_ops")})
    return (out.sort_values(S.FTRANS[:6], kind="stable")[S.FTRANS]
            .reset_index(drop=True))
