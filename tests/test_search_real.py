"""Verifier's checks of pathset/search.py on the real tables of PATHSET_DIR.

Skipped when the network tables (or data-upstream fuas) are missing. Plan,
"Verification", search:
  1. a pure-Python depth-first search that shares no code with search.py
     returns the same candidates (keys and values) as search.candidates at
     MU = inf on the Benelux-Rhine sub-network;
  2. a hop-limited Bellman-Ford over links with minimum transfer times
     (lo + move) and no R0 except the hop limit is never above the least gtt
     of ps-best-by-sequence, and its one-link value equals the least
     one-link gtt there;
  3. MU = inf, then R3 by hand, equals the bounded run (and the stored
     funnel, best-by-sequence and trace of the shard);
  4. the one- and two-link R0 candidates counted with pandas merges equal
     ps-seq-counts.n_r0 (n_r0_bound 0 there);
  5. (findings of 13 Nov) no stored alternative rides through o, d or a city
     of its own path on a link whose passing runs carry PASSED_SHARE of its
     dated departures, and the named pairs' n_r0 is exact.
The references use pathset.transfer (instances, triple_stats) and the
parameters of pathset.io.P only, with PASSED_SHARE of the last enumerate run
in ps-manifest.jsonl (``SHARE``), so that the stored shards and the
references agree. Run time about 2 minutes (one load of the full network,
compile of the search kernels).
"""

import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pathset import io  # noqa: E402
from pathset import transfer as TR  # noqa: E402
from pathset.io import P  # noqa: E402

NEEDED = ["ps-hubs.parquet", "ps-links.parquet", "ps-events.parquet", "ps-event-masks.npy",
          "ps-transfers.parquet", "ps-connectors.parquet", "ps-moves.parquet", "ps-car.parquet"]
pytestmark = pytest.mark.skipif(
    not all((io.OUT / n).exists() for n in NEEDED) or not (io.DATA / "fuas.parquet").exists(),
    reason="real network tables are missing")

# Sub-network of the plan's check: FUAs of NL, BE, LU and German FUAs with
# their centre west of 7.6 E and between 50.3 and 52.0 N
SUB_ICC = ("NL", "BE", "LU")
SUB_DE = (7.6, 50.3, 52.0)
DFS_ORIGINS = [170]           # Leeuwarden: small; 7, 55, 161, 191, 0 were run by hand
FULL_ORIGIN = 133             # Helsinki
FULL_DESTS = [325, 162, 314, 19, 28]
KEY = ["d", "l1", "l2", "l3"]
FLOATS = ["acc", "egr", "mv1", "mv2", "ft1", "ft2", "t_a", "t_dep", "t_ivt", "t_trans",
          "t_tp", "t_arr", "t_e", "T", "gtt", "hwt", "hwt_first", "gtc", "freq", "freq_counted"]
OTHER = ["seq", "n", "h_first", "h_last", "cities", "key", "basis", "n_days"]


def run_share() -> float:
    """PASSED_SHARE of the last enumerate run that ran origins (ps-manifest,
    info "passed_share"); else that of pathset.io.P, 1 without the key."""
    share = float(P.get("PASSED_SHARE", 1.0))
    path = io.OUT / "ps-manifest.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            info = rec.get("info", {})
            if rec.get("step") == "enumerate" and info.get("origins_run"):
                share = float(info.get("passed_share", P.get("PASSED_SHARE", 1.0)))
    return share


SHARE = run_share()
PRM = dict(P, PASSED_SHARE=SHARE)


# --- Tables ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    out = {
        "hubs": io.load_out("ps-hubs", ["hub", "kind", "fua", "iata"]),
        "links": io.load_out("ps-links"),
        "events": io.load_out("ps-events", ["dep", "ride", "trip", "n_dates"]),
        "conn": io.load_out("ps-connectors", ["fua", "hub", "acc_min", "egr_min"]),
        "moves": io.load_out("ps-moves", ["hub_a", "hub_b", "minutes", "kind"]),
        "car": io.load_out("ps-car", ["o", "d", "minutes"]),
        "transfers": io.load_out("ps-transfers", ["link_in", "link_out", "move", "basis",
                                                  "n_days", "freq", "ft_mean"]),
    }
    assert (out["hubs"]["hub"].to_numpy() == np.arange(len(out["hubs"]))).all()
    assert (out["links"]["link"].to_numpy() == np.arange(len(out["links"]))).all()
    out["city"] = np.array(ref_city(out["hubs"], out["conn"]), dtype=np.int64)
    return out


def stored(name: str):
    path = io.OUT / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


# --- Independent definitions (no code of search.py) -----------------------------
def ref_city(hubs, conn) -> list:
    """Station hub: its fua. Airport: fua of its least finite acc_min (ties:
    lower fua); without one, -1 - hub."""
    best = {}
    for f, h, a in zip(conn["fua"], conn["hub"], conn["acc_min"]):
        if np.isfinite(a) and (int(h) not in best or (float(a), int(f)) < best[int(h)]):
            best[int(h)] = (float(a), int(f))
    return [int(f) if k == "S" else (best[int(h)][1] if int(h) in best else -1 - int(h))
            for h, k, f in zip(hubs["hub"], hubs["kind"], hubs["fua"])]


def ref_place(n: int, moves) -> list:
    """Hubs joined by 'colocated' moves are one place (lowest hub)."""
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b, k in zip(moves["hub_a"], moves["hub_b"], moves["kind"]):
        if k == "colocated":
            ra, rb = root(int(a)), root(int(b))
            parent[max(ra, rb)] = min(ra, rb)
    return [root(h) for h in range(n)]


def ref_passed(links, events, city, share=None) -> pd.DataFrame:
    """(link, c): city c is the alight city of another leg of the same run
    (trip key, board hub, departure minute; shorter ride) on the events of
    the link that carry at least ``share`` (default SHARE) of its dated
    departures (ps-events.n_dates), on one at least, and is not the city of
    either hub of the link."""
    share = SHARE if share is None else share
    city = np.asarray(city)
    src, trg = links["src"].to_numpy(), links["trg"].to_numpy()
    n_ev = (links["ev1"] - links["ev0"]).to_numpy()
    link = np.repeat(np.arange(len(links)), n_ev)
    nd = events["n_dates"].to_numpy(np.int64)
    total = np.bincount(link, weights=nd, minlength=len(links))
    E = pd.DataFrame({"e": np.arange(len(events)), "link": link, "dep": events["dep"].to_numpy(),
                      "ride": events["ride"].to_numpy(), "trip": events["trip"].to_numpy(),
                      "src": src[link], "trg": trg[link]})
    parts = []
    for k in range(16):                                    # self-merge by trip chunks
        e = E[E["trip"] % 16 == k]
        M = e.merge(e[["trip", "src", "dep", "ride", "trg"]], on=["trip", "src", "dep"],
                    suffixes=("", "_2"))
        M = M[M["ride_2"] < M["ride"]]
        c = city[M["trg_2"].to_numpy()]
        keep = (c != city[M["src"].to_numpy()]) & (c != city[M["trg"].to_numpy()])
        parts.append(pd.DataFrame({"e": M["e"].to_numpy()[keep],
                                   "link": M["link"].to_numpy()[keep], "c": c[keep]}))
    M = pd.concat(parts, ignore_index=True).drop_duplicates()
    M["w"] = nd[M["e"].to_numpy()]
    cnt = M.groupby(["link", "c"])["w"].sum().reset_index(name="w")
    w = cnt["w"].to_numpy(np.float64)
    keep = (w > 0) & (w >= share * total[cnt["link"].to_numpy()])
    return cnt.loc[keep, ["link", "c"]].sort_values(["link", "c"]).reset_index(drop=True)


def word_ok(word: str) -> bool:
    """MODE_WORDS = air_block_max2: one block of flights, at most two."""
    pos = [i for i, m in enumerate(word) if m == "A"]
    return len(pos) <= 2 and (not pos or pos[-1] - pos[0] == len(pos) - 1)


def exists(basis: str, n_days: int) -> bool:
    return n_days >= P["MIN_ACTIVE_DATES"] if basis == "D" else n_days >= 1


def hidden_wait(f: float) -> float:
    return float(P["HWT_MAX"]) if not f > 0 else min(P["BETA_HWT"] / (2.0 * f), float(P["HWT_MAX"]))


# --- Benelux-Rhine sub-network ----------------------------------------------------
def sub_fuas() -> set:
    f = io.load("fuas", io.DATA, ["id", "icc", "centre"])
    xy = np.array([struct.unpack("<dd" if bytes(b)[0] == 1 else ">dd", bytes(b)[5:21])
                   for b in f["centre"]])
    lon_max, lat_lo, lat_hi = SUB_DE
    keep = f["icc"].isin(SUB_ICC) | ((f["icc"] == "DE") & (xy[:, 0] < lon_max)
                                     & (xy[:, 1] >= lat_lo) & (xy[:, 1] <= lat_hi))
    return set(f.loc[keep, "id"].astype(int))


def benelux_rhine(real) -> dict:
    """Hubs of the FUAs (airports whose city is one of them), links with both
    ends inside, the ps-transfers rows among them, their connectors, moves
    and car rows; hubs and links renumbered, events and masks re-pointed."""
    fset = sub_fuas()
    keep_h = np.flatnonzero(np.isin(real["city"], list(fset)))
    hmap = np.full(len(real["hubs"]), -1, dtype=np.int32)
    hmap[keep_h] = np.arange(len(keep_h))
    lk = real["links"]
    lk = lk[lk["src"].isin(keep_h) & lk["trg"].isin(keep_h)].copy()
    lmap = np.full(len(real["links"]), -1, dtype=np.int32)
    lmap[lk["link"].to_numpy()] = np.arange(len(lk))
    rows = np.concatenate([np.arange(a, b) for a, b in zip(lk["ev0"], lk["ev1"])])
    n_ev = (lk["ev1"] - lk["ev0"]).to_numpy()
    ptr = np.concatenate([[0], np.cumsum(n_ev)])
    events = real["events"].iloc[rows].reset_index(drop=True)
    masks = io.load_npy("ps-event-masks")[rows]
    lk["link"] = np.arange(len(lk), dtype=np.int32)
    lk["src"] = hmap[lk["src"].to_numpy()]
    lk["trg"] = hmap[lk["trg"].to_numpy()]
    lk["ev0"], lk["ev1"] = ptr[:-1].astype(np.int32), ptr[1:].astype(np.int32)
    tr = real["transfers"]
    tr = tr[(lmap[tr["link_in"].to_numpy()] >= 0) & (lmap[tr["link_out"].to_numpy()] >= 0)].copy()
    tr["link_in"] = lmap[tr["link_in"].to_numpy()]
    tr["link_out"] = lmap[tr["link_out"].to_numpy()]
    cn = real["conn"]
    cn = cn[cn["fua"].isin(fset) & cn["hub"].isin(keep_h)].copy()
    cn["hub"] = hmap[cn["hub"].to_numpy()]
    mv = real["moves"]
    mv = mv[mv["hub_a"].isin(keep_h) & mv["hub_b"].isin(keep_h)].copy()
    mv["hub_a"] = hmap[mv["hub_a"].to_numpy()]
    mv["hub_b"] = hmap[mv["hub_b"].to_numpy()]
    car = real["car"]
    hb = real["hubs"].iloc[keep_h].copy()
    hb["hub"] = np.arange(len(hb), dtype=np.int32)
    return dict(hubs=hb.reset_index(drop=True), links=lk.reset_index(drop=True), events=events,
                masks=masks, transfers=tr.reset_index(drop=True),
                connectors=cn.reset_index(drop=True), moves=mv.reset_index(drop=True),
                car=car[car["o"].isin(fset) & car["d"].isin(fset)].reset_index(drop=True))


class DFS:
    """Depth-first enumeration of every R0 candidate of an origin, plain
    Python (search.py module docstring, R0 and Values)."""

    def __init__(self, sub):
        hubs, lk, conn = sub["hubs"], sub["links"], sub["connectors"]
        self.city = ref_city(hubs, conn)
        self.place = ref_place(len(hubs), sub["moves"])
        self.passed = defaultdict(set)
        for l, c in ref_passed(lk, sub["events"], self.city).itertuples(index=False):
            self.passed[int(l)].add(int(c))
        iata = hubs["iata"].fillna("").astype(str)
        self.label = [str(int(f)) if k == "S" else (i or f"A{h}")
                      for h, k, f, i in zip(hubs["hub"], hubs["kind"], hubs["fua"], iata)]
        self.src, self.trg = lk["src"].astype(int).tolist(), lk["trg"].astype(int).tolist()
        self.mode = lk["mode"].tolist()
        self.ride = lk["ride"].to_numpy(np.float32).astype(float).tolist()
        self.freq = lk["freq"].to_numpy(np.float32).astype(float).tolist()
        self.fmax = lk["freq_max"].to_numpy(np.float32).astype(float).tolist()
        self.active = lk["active_dates"].astype(int).tolist()
        self.out = defaultdict(list)
        for l, s in enumerate(self.src):
            self.out[s].append(l)
        t = sub["transfers"]
        self.row, self.succ = {}, defaultdict(list)
        for a, b, mv, bs, nd, fr, ft in zip(t["link_in"], t["link_out"], t["move"].to_numpy(np.float32),
                                            t["basis"], t["n_days"], t["freq"].to_numpy(np.float32),
                                            t["ft_mean"].to_numpy(np.float32)):
            self.row[(int(a), int(b))] = (float(mv), bs, int(nd), float(fr), float(ft))
            self.succ[int(a)].append(int(b))
        self.acc, self.egr = defaultdict(list), defaultdict(list)
        for f, h, a, e in zip(conn["fua"], conn["hub"], conn["acc_min"].to_numpy(np.float32),
                              conn["egr_min"].to_numpy(np.float32)):
            if np.isfinite(a):
                self.acc[int(f)].append((int(h), float(a)))
            if np.isfinite(e):
                self.egr[int(h)].append((int(f), float(e)))
        car = sub["car"]
        self.car = {(int(o), int(d)): float(m) for o, d, m in
                    zip(car["o"], car["d"], car["minutes"].to_numpy(np.float32))}
        ev = sub["events"]
        args = (ev["dep"].to_numpy(), ev["ride"].to_numpy(),
                np.ascontiguousarray(sub["masks"], dtype=np.uint64),
                lk["ev0"].to_numpy(np.int32), lk["ev1"].to_numpy(np.int32),
                lk["tz_shift"].to_numpy(np.int32))
        self.inst, self.typ = TR.build_instances(*args), TR.build_typical(*args)
        self.cal = TR.link_calendars(args[2], args[3], args[4])
        self.trip = ev["trip"].to_numpy(np.int32)
        self.work = TR.workspace(self.inst, self.typ)

    def stops(self, ch):
        s, t = self.src, self.trg
        return ([{s[ch[0]]}] + [{t[ch[k - 1]], s[ch[k]]} for k in range(1, len(ch))]
                + [{t[ch[-1]]}])

    def dead(self, o, ch) -> bool:
        """A clause broken by the prefix that no extension repairs."""
        st = self.stops(ch)
        pl = [{self.place[h] for h in x} for x in st]
        if any(pl[i] & pl[j] for i in range(len(pl)) for j in range(i)):
            return True
        cf = self.city[self.src[ch[0]]]
        tc = [{self.city[h] for h in x} for x in st[1:-1]]
        if any(c & {o, cf} for c in tc) or (len(tc) == 2 and tc[0] & tc[1]):
            return True
        cities = {o} | {self.city[h] for x in st for h in x}
        return any(self.passed[l] & cities for l in ch)

    def chains(self, o):
        for h0, acc in self.acc[o]:
            for l1 in self.out[h0]:
                stack = [[l1]]
                while stack:
                    ch = stack.pop()
                    yield acc, ch
                    if len(ch) <= P["MAX_TRANSFERS"] and not self.dead(o, ch):
                        stack.extend(ch + [b] for b in self.succ[ch[-1]])

    def values(self, ch):
        """(basis, n_days, counted freq, moves, transfer minutes) or None."""
        n = len(ch)
        if n == 1:
            return "D", self.active[ch[0]], self.freq[ch[0]], [], []
        if n == 2:
            mv, basis, n_days, counted, ft = self.row[(ch[0], ch[1])]
            return basis, n_days, counted, [mv], [ft]
        mv = [self.row[(ch[0], ch[1])][0], self.row[(ch[1], ch[2])][0]]
        m = [self.mode[l] for l in ch]
        w = [P["WINDOWS"][("A" if a == "A" else "P") + ("A" if b == "A" else "P")]
             for a, b in ((m[0], m[1]), (m[1], m[2]))]
        typical = (m[0] == "A") != (m[1] == "A") or (m[1] == "A") != (m[2] == "A")
        r = TR.triple_stats(self.inst, self.typ, self.cal, self.trip, ch[0], ch[1], ch[2],
                            mv[0], int(w[0][0]), int(w[0][1]), mv[1], int(w[1][0]), int(w[1][1]),
                            int(P["MIN_COMMON_DATES"]), typical, self.work)
        return ("D" if r[0] == TR.BASIS_D else "T"), int(r[1]), float(r[2]), mv, [float(r[3]), float(r[5])]

    def run(self, o) -> pd.DataFrame:
        rows = []
        for acc, ch in self.chains(o):
            n = len(ch)
            word = "".join(self.mode[l] for l in ch)
            st = self.stops(ch)
            pl = [{self.place[h] for h in x} for x in st]
            h0, hl = self.src[ch[0]], self.trg[ch[-1]]
            cf, cl = self.city[h0], self.city[hl]
            tc = [{self.city[h] for h in x} for x in st[1:-1]]
            seq_h = [h0, self.trg[ch[0]]]
            for k in range(1, n):
                seq_h += ([self.src[ch[k]]] if self.src[ch[k]] != seq_h[-1] else []) + [self.trg[ch[k]]]
            board = set().union(*(self.passed[l] for l in ch))
            if (not word_ok(word) or any(pl[i] & pl[j] for i in range(len(pl)) for j in range(i))
                    or cf == cl or cl == o or any(c & {o, cf, cl} for c in tc)
                    or (n == 3 and tc[0] & tc[1])
                    or board & ({o} | {self.city[h] for h in seq_h})):
                continue
            basis, n_days, counted, mv, ft = self.values(ch)
            if not exists(basis, n_days):
                continue
            freq = min(counted, min(self.fmax[l] for l in ch))
            m1, ml = self.mode[ch[0]], self.mode[ch[-1]]
            toks = [self.label[h] for h in seq_h]
            pattern = ">".join(x for i, x in enumerate(toks) if i == 0 or x != toks[i - 1])
            for d, egr in self.egr[hl]:
                if d == o or d == cf or d in board or any(d in c for c in tc):
                    continue
                T = acc + P["TAU_DEP"][m1] + sum(self.ride[l] for l in ch) + sum(ft) + P["TAU_ARR"][ml] + egr
                if not T <= P["T_MAX_MIN"]:
                    continue
                comp = dict(t_a=P["BETA_ACCESS"] * acc, t_dep=P["TAU_DEP"][m1],
                            t_ivt=sum(P["BETA_IVT"][self.mode[l]] * self.ride[l] for l in ch),
                            t_trans=P["BETA_TRANS"] * sum(ft), t_tp=P["TAU_TP"] * (n - 1),
                            t_arr=P["TAU_ARR"][ml], t_e=P["BETA_EGRESS"] * egr)
                gtt = sum(comp.values())
                pad = ch + [-1] * (3 - n)
                rows.append(dict(d=d, l1=pad[0], l2=pad[1], l3=pad[2], seq=word, n=n - 1,
                                 h_first=h0, h_last=hl, cities=pattern, key="-".join(map(str, ch)),
                                 acc=acc, egr=egr, mv1=(mv + [np.nan] * 2)[0],
                                 mv2=(mv + [np.nan] * 2)[1], ft1=(ft + [np.nan] * 2)[0],
                                 ft2=(ft + [np.nan] * 2)[1], **comp, T=T, gtt=gtt,
                                 hwt=hidden_wait(freq), hwt_first=hidden_wait(self.freq[ch[0]]),
                                 gtc=gtt + hidden_wait(freq), freq=freq, freq_counted=counted,
                                 basis=basis, n_days=n_days))
        if P["INCLUDE_CAR"]:
            for (a, d), m in self.car.items():
                if a == o and d != o:
                    g = P["BETA_IVT"]["C"] * m
                    rows.append(dict(d=d, l1=-1, l2=-1, l3=-1, seq="C", n=0, h_first=-1,
                                     h_last=-1, cities="", key="car", acc=0.0, egr=0.0,
                                     mv1=np.nan, mv2=np.nan, ft1=np.nan, ft2=np.nan, t_a=0.0,
                                     t_dep=0.0, t_ivt=g, t_trans=0.0, t_tp=0.0, t_arr=0.0,
                                     t_e=0.0, T=m, gtt=g, hwt=0.0, hwt_first=0.0, gtc=g,
                                     freq=np.inf, freq_counted=np.inf, basis="", n_days=0))
        return pd.DataFrame(rows)


def assert_same_rows(got: pd.DataFrame, want: pd.DataFrame, tol: float) -> None:
    """Same multiset of (d, l1, l2, l3), and the same values (floats within
    tol, NaN = NaN, inf = inf)."""
    for df in (got, want):
        assert not df.duplicated(KEY).any()
    m = got.astype({k: "int64" for k in KEY}).merge(
        want.astype({k: "int64" for k in KEY}), on=KEY, how="outer",
        suffixes=("_g", "_w"), indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts().to_dict()
    for c in FLOATS:
        a, b = m[c + "_g"].to_numpy(np.float64), m[c + "_w"].to_numpy(np.float64)
        with np.errstate(invalid="ignore"):
            ok = (np.abs(a - b) <= tol) | (np.isnan(a) & np.isnan(b)) | (a == b)
        assert ok.all(), (c, int((~ok).sum()))
    for c in OTHER:
        bad = m[c + "_g"].astype(str) != m[c + "_w"].astype(str)
        assert not bad.any(), (c, int(bad.sum()))


# --- Tests ----------------------------------------------------------------------------
def test_cities_places_and_passed_cities_on_the_full_network(real):
    from pathset import search as SR
    hubs, lk, ev = real["hubs"], real["links"], real["events"]
    assert np.array_equal(SR.hub_cities(hubs, real["conn"]), real["city"])
    assert np.array_equal(SR.hub_places(len(hubs), real["moves"]),
                          ref_place(len(hubs), real["moves"]))
    dates = SR.event_dates(io.load_npy("ps-event-masks"))
    assert np.array_equal(dates, ev["n_dates"].to_numpy(np.int64))
    for share in sorted({SHARE, 1.0}):
        ptr, cities = SR.passed_cities(lk["src"].to_numpy(), lk["trg"].to_numpy(),
                                       lk["ev0"].to_numpy(), lk["ev1"].to_numpy(),
                                       ev["dep"].to_numpy(), ev["ride"].to_numpy(),
                                       ev["trip"].to_numpy(), real["city"], dates, share)
        theirs = pd.DataFrame({"link": np.repeat(np.arange(len(lk)), np.diff(ptr)), "c": cities})
        mine = ref_passed(lk, ev, real["city"], share)
        assert len(mine) > 0
        pd.testing.assert_frame_equal(theirs.sort_values(["link", "c"]).reset_index(drop=True),
                                      mine.astype("int64"), check_dtype=False)


def test_stored_alternatives_never_ride_through_their_cities(real):
    """Finding of 13 Nov: no stored service alternative uses a link that
    passes (at SHARE) o, d or the city of a hub of its own path; e.g.
    Amsterdam -> Koeln no longer rides D-Flughafen -> Bonn through Koeln."""
    alt = stored("ps-alternatives")
    if alt is None or not len(alt):
        pytest.skip("no stored ps-alternatives")
    alt = alt[alt["seq"] != "C"]
    city, lk = real["city"], real["links"]
    src, trg = lk["src"].to_numpy(), lk["trg"].to_numpy()
    pas = ref_passed(lk, real["events"], city)
    by_link = pas.groupby("link")["c"].apply(set).to_dict()
    bad = []
    for r in alt.itertuples():
        ls = [int(x) for x in (r.l1, r.l2, r.l3) if x >= 0]
        cities = {int(r.o), int(r.d)} | {int(city[src[l]]) for l in ls} | {int(city[trg[l]]) for l in ls}
        if any(by_link.get(l, set()) & cities for l in ls):
            bad.append((r.o, r.d, r.rank))
    assert not bad, bad[:10]


@pytest.mark.parametrize("o", DFS_ORIGINS)
def test_depth_first_search_equals_search_on_benelux_rhine(real, o):
    from pathset import search as SR
    sub = benelux_rhine(real)
    net = SR.build_network(sub["hubs"], sub["links"], sub["events"], sub["masks"],
                           sub["transfers"], sub["connectors"], sub["car"], PRM, sub["moves"])
    want = DFS(sub).run(o)
    got = SR.candidates(net, o, None, PRM, mu=math.inf)
    assert (want["l3"] >= 0).any() and (want["seq"] == "C").any()
    assert_same_rows(got, want, 1e-3)


def one_two_link_counts(real, o) -> dict:
    """R0 candidates of one and two links per mode sequence, by merges."""
    city, conn, lk = real["city"], real["conn"], real["links"]
    place = np.array(ref_place(len(real["hubs"]), real["moves"]))
    pas = ref_passed(lk, real["events"], city)
    base = int(city.max() - city.min()) + 1
    keys = pas["link"].to_numpy(np.int64) * base + (pas["c"].to_numpy(np.int64) - city.min())

    def passes(l, c):
        return np.isin(l.astype(np.int64) * base + (c.astype(np.int64) - city.min()), keys)

    acc = conn[(conn["fua"] == o) & np.isfinite(conn["acc_min"])][["hub", "acc_min"]]
    acc.columns = ["h0", "acc"]
    eg = conn[np.isfinite(conn["egr_min"])][["hub", "fua", "egr_min"]]
    eg.columns = ["hl", "d", "egr"]
    L = lk[["link", "src", "trg", "mode", "ride"]].astype({"ride": "float64"})
    dep = {k: float(v) for k, v in P["TAU_DEP"].items()}
    arr = {k: float(v) for k, v in P["TAU_ARR"].items()}
    one = acc.merge(L.rename(columns={"link": "l1", "src": "h0", "trg": "h1", "mode": "m1",
                                      "ride": "r1"}), on="h0")
    x = one.merge(eg, left_on="h1", right_on="hl")
    cf, cl, d = city[x["h0"]], city[x["h1"]], x["d"].to_numpy()
    T = (x["acc"].astype("float64") + x["m1"].map(dep) + x["r1"] + x["m1"].map(arr)
         + x["egr"].astype("float64")).to_numpy()
    ok = ((d != o) & (cf != cl) & (cf != d) & (cl != o) & (place[x["h0"]] != place[x["h1"]])
          & (T <= P["T_MAX_MIN"]))
    for c in (np.full(len(x), o), d, cf, cl):
        ok &= ~passes(x["l1"].to_numpy(), c)
    out = x[ok].groupby("m1").size().to_dict()
    tr = real["transfers"]
    tr = tr[tr["link_in"].isin(one["l1"])]
    tr = tr[[exists(b, n) for b, n in zip(tr["basis"], tr["n_days"])]]
    two = one.merge(tr[["link_in", "link_out", "ft_mean"]], left_on="l1", right_on="link_in")
    two = two.merge(L.rename(columns={"link": "link_out", "src": "h1b", "trg": "h2", "mode": "m2",
                                      "ride": "r2"}), on="link_out")
    x = two.merge(eg, left_on="h2", right_on="hl")
    h0, h1, h1b, h2 = (x[k].to_numpy() for k in ("h0", "h1", "h1b", "h2"))
    cf, c1, c1b, cl, d = city[h0], city[h1], city[h1b], city[h2], x["d"].to_numpy()
    T = (x["acc"].astype("float64") + x["m1"].map(dep) + x["r1"] + x["ft_mean"].astype("float64")
         + x["r2"] + x["m2"].map(arr) + x["egr"].astype("float64")).to_numpy()
    ok = ((d != o) & (cf != cl) & (cl != o) & (d != cf) & (T <= P["T_MAX_MIN"])
          & (place[h0] != place[h1]) & (place[h0] != place[h1b]) & (place[h2] != place[h0])
          & (place[h2] != place[h1]) & (place[h2] != place[h1b]))
    for c in (c1, c1b):
        ok &= (c != o) & (c != d) & (c != cf) & (c != cl)
    for c in (np.full(len(x), o), d, cf, c1, c1b, cl):
        ok &= ~passes(x["l1"].to_numpy(), c) & ~passes(x["link_out"].to_numpy(), c)
    out.update((x["m1"] + x["m2"])[ok].value_counts().to_dict())
    return {str(k): int(v) for k, v in out.items()}


@pytest.mark.parametrize("o", [7, 133])
def test_one_and_two_link_counts_equal_seq_counts(real, o):
    sc = stored("ps-seq-counts")
    if sc is None or not (sc["o"] == o).any():
        pytest.skip(f"no ps-seq-counts rows of origin {o}")
    sc = sc[(sc["o"] == o) & (sc["seq"].str.len() <= 2) & (sc["seq"] != "C")].set_index("seq")
    mine = one_two_link_counts(real, o)
    assert (sc["n_r0_bound"] == 0).all()
    assert {s: int(v) for s, v in sc["n_r0"].items() if v} == mine


def bellman_ford(real, o, hops=3) -> pd.DataFrame:
    """Least relaxed gtt per destination over paths of exactly k links (column
    h<k>): minimum transfer time lo + move on every ps-transfers row, no R0."""
    lk, conn, tr = real["links"], real["conn"], real["transfers"]
    mc = {"A": 0, "B": 1, "R": 2}
    m = lk["mode"].map(mc).to_numpy()
    ride = lk["ride"].to_numpy(np.float64)
    src, trg = lk["src"].to_numpy(), lk["trg"].to_numpy()
    ivt, dep, arr = (np.array([P[k][x] for x in "ABR"]) for k in ("BETA_IVT", "TAU_DEP", "TAU_ARR"))
    lo = np.array([[P["WINDOWS"][("A" if a == "A" else "P") + ("A" if b == "A" else "P")][0]
                    for b in "ABR"] for a in "ABR"], dtype=np.float64)
    a = conn[(conn["fua"] == o) & np.isfinite(conn["acc_min"])]
    acc = np.full(len(real["hubs"]), np.inf)
    acc[a["hub"].to_numpy()] = a["acc_min"].to_numpy(np.float64)
    cur = P["BETA_ACCESS"] * acc[src] + dep[m] + ivt[m] * ride
    li, lo_ = tr["link_in"].to_numpy(), tr["link_out"].to_numpy()
    step = (P["BETA_TRANS"] * (lo[m[li], m[lo_]] + tr["move"].to_numpy(np.float64))
            + P["TAU_TP"] + ivt[m[lo_]] * ride[lo_])
    eg = conn[np.isfinite(conn["egr_min"]) & (conn["fua"] != o)]
    cols = []
    for k in range(1, hops + 1):
        if k > 1:
            nxt = np.full(len(lk), np.inf)
            np.minimum.at(nxt, lo_, cur[li] + step)
            cur = nxt
        end = pd.Series(cur + arr[m]).groupby(trg).min()
        v = eg["egr_min"].to_numpy(np.float64) * P["BETA_EGRESS"] + end.reindex(eg["hub"]).to_numpy()
        cols.append(pd.Series(v, index=eg["fua"].to_numpy()).groupby(level=0).min().rename(f"h{k}"))
    return pd.concat(cols, axis=1)


@pytest.mark.parametrize("o", [7, 133])
def test_bellman_ford_never_above_best_by_sequence(real, o):
    best = stored("ps-best-by-sequence")
    if best is None or not (best["o"] == o).any():
        pytest.skip(f"no ps-best-by-sequence rows of origin {o}")
    b = best[(best["o"] == o) & (best["seq"] != "C")]
    bf = bellman_ford(real, o)
    least = b.groupby("d")["gtt"].min().astype(np.float64)
    low = bf.min(axis=1).reindex(least.index)
    assert (low <= least + 1e-3).all()
    one = b[b["n"] == 0].groupby("d")["gtt"].min().astype(np.float64)
    assert np.allclose(bf["h1"].reindex(one.index), one, atol=1e-3, rtol=0)


@pytest.fixture(scope="module")
def full_133():
    """MU = inf and the bounded run of FULL_ORIGIN on FULL_DESTS (full network)."""
    from pathset import search as SR
    net = SR.load_network(PRM)
    full = SR.candidates(net, FULL_ORIGIN, FULL_DESTS, PRM, mu=math.inf)
    bounded = SR.candidates(net, FULL_ORIGIN, FULL_DESTS, PRM, mu=float(P["MU"]))
    return full, bounded


def r3_by_hand(full: pd.DataFrame, mu: float) -> pd.DataFrame:
    """Cut iff gtt > (1 + mu) x least gtt AND gtc > (1 + mu) x least gtc of its
    mode class (car its own class), per destination, on stored float32."""
    cls = full["seq"].map(lambda s: "C" if s == "C" else "".join(sorted(set(s))))
    g, h = (full[c].astype(np.float32).astype(np.float64) for c in ("gtt", "gtc"))
    mg = g.groupby([full["d"], cls]).transform("min")
    mh = h.groupby([full["d"], cls]).transform("min")
    return full[((g <= (1 + mu) * mg) | (h <= (1 + mu) * mh)).to_numpy()]


def test_r3_by_hand_after_mu_inf_equals_bounded_run(full_133):
    full, bounded = full_133
    hand = r3_by_hand(full, float(P["MU"]))
    assert len(hand) < len(full)
    assert_same_rows(bounded, hand, 0.0)


def test_stored_shard_agrees_with_mu_inf(full_133):
    full, bounded = full_133
    fun = stored("ps-funnel")
    if fun is None or not (fun["o"] == FULL_ORIGIN).any():
        pytest.skip(f"no stored shard of origin {FULL_ORIGIN}")
    fun = fun[(fun["o"] == FULL_ORIGIN) & fun["d"].isin(FULL_DESTS)].set_index("d")
    exact = full.groupby("d").size().reindex(fun.index, fill_value=0)
    assert ((fun["n_r0"] - fun["n_r0_bound"] <= exact) & (exact <= fun["n_r0"])).all()
    trace = stored("ps-trace")
    traced = [d for d in FULL_DESTS if trace is not None
              and ((trace["o"] == FULL_ORIGIN) & (trace["d"] == d)).any()]
    if traced:                                   # named pairs: n_r0 exact
        assert (fun.loc[traced, "n_r0"] == exact.loc[traced]).all()
        assert (fun.loc[traced, "n_r0_bound"] == 0).all()
    assert (bounded.groupby("d").size().reindex(fun.index, fill_value=0) == fun["n_r3"]).all()
    best = stored("ps-best-by-sequence")
    best = best[(best["o"] == FULL_ORIGIN) & best["d"].isin(FULL_DESTS)]
    mine = (full.assign(_g=full["gtt"].astype(np.float32))
            .sort_values(["d", "seq", "_g", "l1", "l2", "l3"])
            .drop_duplicates(["d", "seq"]).drop(columns="_g"))
    assert_same_rows(best, mine, 1e-3)
    for d in FULL_DESTS:
        t = trace[(trace["o"] == FULL_ORIGIN) & (trace["d"] == d)] if trace is not None else []
        if len(t):
            assert_same_rows(t, full[full["d"] == d], 1e-3)
