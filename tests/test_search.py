"""Path search (pathset/search.py) on small synthetic networks.

The networks are built here (``Toy``): hubs, links with events and calendars,
moves, connectors and car rows by hand; ps-links computed directly; ps-transfers
with pathset.transfer.transfer_table on every same-hub and move pair with the
plan's windows (air <-> ground on typical days). The reference ``brute`` is
pure Python and shares no code with search.py: it walks the tables, applies R0
clause by clause and takes the values from pathset.cost and
pathset.transfer.triple_stats.
"""

import copy
import importlib.util
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pathset import cost as C  # noqa: E402
from pathset import dates as D  # noqa: E402
from pathset import io  # noqa: E402
from pathset import schemas as S  # noqa: E402
from pathset import search as SR  # noqa: E402
from pathset import transfer as TR  # noqa: E402
from pathset.io import P  # noqa: E402

DAYS = tuple(range(100, 130))            # 30 dates
MONDAYS = tuple(d for d in range(100, 170) if D.index_date(d).weekday() == 0)
WINDOW = {k: tuple(v) for k, v in P["WINDOWS"].items()}


def hm(h, m=0):
    return 60 * h + m


def params(**changes):
    out = copy.deepcopy(P)
    out.update(changes)
    return out


# --- Synthetic networks --------------------------------------------------------
class Toy:
    """A synthetic network in the tables of the contract. Hubs are numbered in
    order of creation, links too; every event is its own trip."""

    def __init__(self, days=DAYS):
        self.days = tuple(days)
        self.hubs, self.links, self.conn, self.car = [], [], [], []
        self.moves = {}
        self._trip = 0

    def hub(self, fua=-1, iata=""):
        """A station hub of FUA ``fua``, or an airport when ``iata`` is given."""
        self.hubs.append(("A" if iata else "S", -1 if iata else fua, iata))
        return len(self.hubs) - 1

    def link(self, src, trg, mode, *events):
        """A service link; an event is (dep, ride) or (dep, ride, days)."""
        evs = []
        for e in events:
            self._trip += 1
            evs.append((e[0], e[1], self._trip, tuple(e[2] if len(e) > 2 else self.days)))
        self.links.append((src, trg, mode, evs))
        return len(self.links) - 1

    def move(self, a, b, minutes):
        self.moves[(a, b)] = self.moves[(b, a)] = float(minutes)

    def connect(self, fua, hub, acc, egr=None):
        self.conn.append((fua, hub, acc, acc if egr is None else egr))

    def drive(self, o, d, minutes):
        self.car.append((o, d, minutes))

    def tables(self) -> dict:
        """The contract tables, and the instances that built ps-transfers."""
        n_hubs = len(self.hubs)
        hubs = pd.DataFrame({
            "hub": np.arange(n_hubs, dtype=np.int32),
            "kind": [h[0] for h in self.hubs], "name": [f"hub {i}" for i in range(n_hubs)],
            "fua": np.array([h[1] for h in self.hubs], dtype=np.int16), "icc": "NL",
            "tz_std": np.int8(1), "lon": 5.0, "lat": 52.0,
            "iata": [h[2] for h in self.hubs], "stns": [[] for _ in range(n_hubs)],
            "released_id": "", "match_m": np.float32(np.nan)})
        ev_rows, active_rows, link_rows = [], [], []
        classes = defaultdict(int)
        for l, (src, trg, mode, evs) in enumerate(self.links):
            evs = sorted(evs)
            ev0 = len(ev_rows)
            count = np.zeros(D.N_DAYS, dtype=np.int64)
            for dep, ride, trip, days in evs:
                ev_rows.append((l, dep, ride, trip, len(set(days))))
                row = np.zeros(D.N_DAYS, dtype=bool)
                row[list(days)] = True
                active_rows.append(row)
                count += row
            pos = count[count > 0]
            rides = np.array([e[1] for e in evs], dtype=float)
            weights = np.array([len(set(e[3])) for e in evs], dtype=float)
            link_rows.append(dict(
                link=l, src=src, trg=trg, mode=mode, cls=classes[(src, trg, mode)],
                ride=(rides * weights).sum() / weights.sum(), ride_min=rides.min(),
                ride_max=rides.max(), tz_shift=0, km=100.0, ev0=ev0, ev1=len(ev_rows),
                n_events=len(evs), active_dates=len(pos),
                first_day=np.flatnonzero(count)[0], last_day=np.flatnonzero(count)[-1],
                freq=np.median(pos), freq_max=pos.max(), freq_span=0.0, freq_fixed=0.0,
                active_share=1.0))
            classes[(src, trg, mode)] += 1
        links = pd.DataFrame(link_rows)[S.LINKS].astype({
            "link": "int32", "src": "int32", "trg": "int32", "cls": "int16",
            "ride": "float32", "ride_min": "int16", "ride_max": "int16",
            "tz_shift": "int16", "km": "float32", "ev0": "int32", "ev1": "int32",
            "n_events": "int32", "active_dates": "int32", "first_day": "int32",
            "last_day": "int32", "freq": "float32", "freq_max": "int16",
            "freq_span": "float32", "freq_fixed": "float32", "active_share": "float32"})
        events = pd.DataFrame(ev_rows, columns=S.EVENTS).astype({
            "link": "int32", "dep": "int16", "ride": "int16", "trip": "int32",
            "n_dates": "int32"})
        masks = D.pack_bool(np.array(active_rows))
        args = (events["dep"].to_numpy(), events["ride"].to_numpy(), masks,
                links["ev0"].to_numpy(), links["ev1"].to_numpy(),
                links["tz_shift"].to_numpy())
        inst, typ = TR.build_instances(*args), TR.build_typical(*args)
        cal = TR.link_calendars(masks, links["ev0"].to_numpy(np.int32),
                                links["ev1"].to_numpy(np.int32))
        trip = events["trip"].to_numpy(np.int32)
        connectors = pd.DataFrame(self.conn, columns=["fua", "hub", "acc_min", "egr_min"])
        connectors["source"] = "fitted"
        connectors = connectors.astype({"fua": "int16", "hub": "int32",
                                        "acc_min": "float32", "egr_min": "float32"})
        car = pd.DataFrame(self.car, columns=["o", "d", "minutes"]).astype(
            {"o": "int16", "d": "int16", "minutes": "float32"})
        car["km"] = np.float32(100)
        car["fixed"] = False
        return dict(hubs=hubs, links=links, events=events, masks=masks,
                    transfers=self.transfer_rows(links, inst, typ, cal, trip),
                    connectors=connectors, car=car,
                    inst=inst, typ=typ, cal=cal, trip=trip)

    def transfer_rows(self, links, inst, typ, cal, trip) -> pd.DataFrame:
        """ps-transfers: every pair (link_in, link_out) at one hub or across
        a move, counted by transfer.transfer_table; rows with an opportunity."""
        by_src = defaultdict(list)
        for l, (src, *_rest) in enumerate(self.links):
            by_src[src].append(l)
        moves_from = defaultdict(list)
        for (a, b), m in self.moves.items():
            moves_from[a].append((b, m))
        rows = []
        for li, (_, trg, m_in, _) in enumerate(self.links):
            for hub, move in [(trg, 0.0)] + moves_from[trg]:
                for lo in by_src[hub]:
                    if lo != li:
                        m_out = self.links[lo][2]
                        rows.append((li, lo, move, *WINDOW[S.window_key(m_in, m_out)],
                                     (m_in == "A") != (m_out == "A")))
        cols = list(zip(*rows)) if rows else [[]] * 6
        out = TR.transfer_table(inst, typ, cal, trip, *[np.array(c) for c in cols[:5]],
                                typical=np.array(cols[5], dtype=bool))
        df = pd.DataFrame({"link_in": np.array(cols[0], dtype=np.int32),
                           "link_out": np.array(cols[1], dtype=np.int32),
                           "move": np.array(cols[2], dtype=np.float32),
                           "basis": TR.basis_labels(out["basis"]),
                           "n_days": out["n_days"], "freq": out["freq"],
                           "ft_mean": out["ft_mean"], "ft_min": out["ft_min"]})
        return df[df["n_days"] >= 1].reset_index(drop=True)


def net_of(tb: dict, prm=P) -> SR.Net:
    return SR.build_network(tb["hubs"], tb["links"], tb["events"], tb["masks"],
                            tb["transfers"], tb["connectors"], tb["car"], prm)


def random_toy(seed: int, n_fua=6, n_ground=40, n_air=6) -> Toy:
    """FUAs 10.. with one or two stations (two: a move of 8 min), stations that
    also serve a neighbouring FUA, three airports (one serves no FUA and is its
    own city), rail and coach links with three to six runs on daily, weekday
    or thin calendars (some pairs share fewer than 14 dates), flights, cars."""
    rng = np.random.default_rng(seed)
    t = Toy()
    fuas = list(range(10, 10 + n_fua))
    stations = []
    for f in fuas:
        hs = [t.hub(f)]
        if rng.random() < 0.5:
            hs.append(t.hub(f))
            t.move(hs[0], hs[1], 8)
        for h in hs:
            t.connect(f, h, int(rng.integers(5, 26)), int(rng.integers(5, 26)))
        stations += hs
    for f, g in zip(fuas, fuas[1:]):
        if rng.random() < 0.5:
            t.connect(g, stations[fuas.index(f)], 40, 45)
    airports = [t.hub(iata=c) for c in ("AAA", "BBB", "CCC")]
    t.connect(fuas[0], airports[0], 30)
    t.connect(fuas[1], airports[0], 60)
    t.connect(fuas[3], airports[1], 35)
    t.connect(fuas[4], airports[1], 35, 50)
    t.move(airports[0], stations[0], 20)
    t.move(airports[1], stations[-1], 25)
    calendars = [DAYS, DAYS, tuple(d for d in DAYS if D.index_date(d).weekday() < 5),
                 tuple(range(100, 112)), tuple(range(118, 130))]
    for _ in range(n_ground):
        a, b = rng.choice(len(stations), 2, replace=False)
        days = calendars[int(rng.integers(0, len(calendars)))]
        ride = int(rng.integers(6, 49)) * 5
        t.link(int(stations[a]), int(stations[b]), "RB"[int(rng.integers(0, 2))],
               *[(int(rng.integers(5 * 12, 21 * 12)) * 5, ride + int(rng.choice([0, 0, 10])),
                  days) for _ in range(int(rng.integers(3, 7)))])
    for _ in range(n_air):
        a, b = rng.choice(3, 2, replace=False)
        t.link(airports[a], airports[b], "A",
               *[(int(rng.integers(6 * 12, 20 * 12)) * 5, int(rng.integers(12, 30)) * 5)
                 for _ in range(int(rng.integers(1, 4)))])
    for o in fuas:
        for d in fuas:
            if o != d and rng.random() < 0.6:
                t.drive(o, d, float(rng.integers(60, 600)))
    return t


# --- Brute-force reference -----------------------------------------------------
def ref_word_ok(word: str, rule: str) -> bool:
    runs = [k for k, _ in itertools.groupby(word)]
    ok = runs.count("A") <= 1 and word.count("A") <= 2
    if rule == "deck35":
        return ok and word != "RAR"
    if rule == "oct21":
        return ok and word not in {"ABB", "ABR", "ARB", "ARR", "BBA", "BRA", "RBA", "RRA"}
    return ok


def ref_cities(tb: dict) -> dict:
    hubs, conn = tb["hubs"], tb["connectors"]
    city = {}
    for h, kind, fua in zip(hubs["hub"], hubs["kind"], hubs["fua"]):
        if kind == "S":
            city[int(h)] = int(fua)
            continue
        rows = sorted((float(a), int(f)) for f, hh, a in
                      zip(conn["fua"], conn["hub"], conn["acc_min"])
                      if hh == h and np.isfinite(a))
        city[int(h)] = rows[0][1] if rows else -1 - int(h)
    return city


def brute(tb: dict, o: int, prm=P) -> dict:
    """Every R0 candidate of origin o: {(d, l1, l2, l3): values}."""
    links, conn = tb["links"], tb["connectors"]
    city = ref_cities(tb)
    labels = {int(h): (str(int(f)) if k == "S" else i) for h, k, f, i in zip(
        tb["hubs"]["hub"], tb["hubs"]["kind"], tb["hubs"]["fua"], tb["hubs"]["iata"])}
    access = {int(h): float(a) for f, h, a in zip(conn["fua"], conn["hub"], conn["acc_min"])
              if f == o and np.isfinite(a)}
    egress = defaultdict(list)
    for f, h, e in zip(conn["fua"], conn["hub"], conn["egr_min"]):
        if np.isfinite(e):
            egress[int(h)].append((int(f), float(e)))
    src, trg = links["src"].tolist(), links["trg"].tolist()
    mode, ride = links["mode"].tolist(), links["ride"].tolist()
    freq, fmax, active = links["freq"].tolist(), links["freq_max"].tolist(), links["active_dates"].tolist()
    rows = {(int(r.link_in), int(r.link_out)): r for r in tb["transfers"].itertuples()}
    succ = defaultdict(list)
    for a, b in rows:
        succ[a].append(b)
    chains = []

    def grow(chain):
        chains.append(tuple(chain))
        if len(chain) <= prm["MAX_TRANSFERS"]:
            for b in succ[chain[-1]]:
                grow(chain + [b])

    for l in range(len(src)):
        if src[l] in access:
            grow([l])
    work = TR.workspace(tb["inst"], tb["typ"])
    par = C.param_array(prm)
    min_active = prm["MIN_ACTIVE_DATES"]
    out = {}
    for chain in chains:
        word = "".join(mode[l] for l in chain)
        if not ref_word_ok(word, prm["MODE_WORDS"]):
            continue
        n = len(chain)
        mv = [float(rows[chain[k], chain[k + 1]].move) for k in range(n - 1)]
        if n == 1:
            basis, n_days, counted, ft = "D", active[chain[0]], freq[chain[0]], []
        elif n == 2:
            r = rows[chain]
            basis, n_days, counted, ft = r.basis, r.n_days, r.freq, [r.ft_mean]
        else:
            m = [mode[l] for l in chain]
            forced = (m[0] == "A") != (m[1] == "A") or (m[1] == "A") != (m[2] == "A")
            w1, w2 = WINDOW[S.window_key(m[0], m[1])], WINDOW[S.window_key(m[1], m[2])]
            res = TR.triple_stats(tb["inst"], tb["typ"], tb["cal"], tb["trip"], *chain,
                                  mv[0], w1[0], w1[1], mv[1], w2[0], w2[1],
                                  prm["MIN_COMMON_DATES"], forced, work)
            basis, n_days, counted, ft = chr(res[0]), res[1], res[2], [res[3], res[5]]
        if not (n_days >= min_active if basis == "D" else n_days >= 1):
            continue
        # hubs in order and the cities of each transfer
        seq = [src[chain[0]], trg[chain[0]]]
        tcity = []
        for k in range(1, n):
            a, b = trg[chain[k - 1]], src[chain[k]]
            if b != a:
                seq.append(b)
            seq.append(trg[chain[k]])
            tcity.append({city[a], city[b]})
        if len(set(seq)) != len(seq):
            continue
        cf, cl = city[seq[0]], city[seq[-1]]
        if n == 3 and tcity[0] & tcity[1]:
            continue
        acc = access[seq[0]]
        for d, egr in egress[seq[-1]]:
            if o == d or cf == cl or cf == d or cl == o:
                continue
            if any(tc & {o, d, cf, cl} for tc in tcity):
                continue
            pad = list(chain) + [-1] * (3 - n)
            capped = C.capped_frequency([n], [counted], [[fmax[l] if l >= 0 else 0 for l in pad]])
            cost = C.path_costs([n], [acc], [[C.MODE_CODE[mode[l]] if l >= 0 else 0 for l in pad]],
                                [[ride[l] if l >= 0 else 0.0 for l in pad]],
                                [(ft + [0.0, 0.0])[:2]], [egr], capped, par)
            if not cost["T"][0] <= prm["T_MAX_MIN"]:
                continue
            text = [labels[h] for h in seq]
            out[(d, *pad)] = dict(
                seq=word, n=n - 1, h_first=seq[0], h_last=seq[-1],
                cities=">".join(k for k, _ in itertools.groupby(text)),
                key="-".join(map(str, chain)), acc=acc, egr=egr,
                mv1=(mv + [np.nan, np.nan])[0], mv2=(mv + [np.nan, np.nan])[1],
                ft1=(ft + [np.nan, np.nan])[0], ft2=(ft + [np.nan, np.nan])[1],
                **{k: cost[k][0] for k in C.COMPONENTS + ("T", "gtt", "hwt", "gtc")},
                hwt_first=C.hidden_wait([freq[chain[0]]], par)[0], freq=capped[0],
                freq_counted=counted, basis=basis, n_days=n_days)
    return out


def service(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["seq"] != "C"]


def as_dict(df: pd.DataFrame) -> dict:
    """{(d, l1, l2, l3): row} of candidate rows."""
    keys = zip(df["d"], df["l1"], df["l2"], df["l3"])
    return {tuple(int(x) for x in k): r for k, r in zip(keys, df.to_dict("records"))}


def same_rows(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want)
    for k, w in want.items():
        g = got[k]
        for col, v in w.items():
            if isinstance(v, str):
                assert g[col] == v, (k, col)
            else:
                assert np.float32(g[col]) == np.float32(v) or (
                    np.isnan(np.float32(v)) and np.isnan(g[col])), (k, col, g[col], v)


def present(net, o, prm=P) -> set:
    """(d, link ids ...) of the service candidates of origin o."""
    c = service(SR.candidates(net, o, None, prm))
    return {(int(d), *[int(x) for x in ls if x >= 0])
            for d, *ls in zip(c["d"], c["l1"], c["l2"], c["l3"])}


@pytest.fixture(scope="module")
def rand():
    """Two random networks as (tables, net)."""
    out = []
    for seed in (1, 2):
        tb = random_toy(seed).tables()
        out.append((tb, net_of(tb)))
    return out


# --- 1. R0, clause by clause ---------------------------------------------------
HOURLY = [(hm(h), 40) for h in range(6, 23)]     # arrive at :40, leave at :00


def clause_toy():
    """FUAs O = 1, N = 2, X = 3, Y = 4, D = 5; hourly rail links that connect
    at every hub. O has access at o1, at n1 (city N) and at dd (city D)."""
    t = Toy()
    o1, o2, n1, n2 = t.hub(1), t.hub(1), t.hub(2), t.hub(2)
    x1, x2, y1, d1, dd = t.hub(3), t.hub(3), t.hub(4), t.hub(5), t.hub(5)
    t.connect(1, o1, 10)
    t.connect(1, n1, 30)
    t.connect(1, dd, 50)
    for h in (d1, n2, x1, o2, x2):
        t.connect(5, h, 20)
    t.connect(3, x1, 10)
    hubs = dict(o1=o1, o2=o2, n1=n1, n2=n2, x1=x1, x2=x2, y1=y1, d1=d1, dd=dd)
    L = {}
    for pair in ["n1 x1", "x1 n2", "x1 d1", "dd x1", "n1 o2", "n1 o1", "o1 x1",
                 "o1 d1", "d1 x1", "n1 n2", "n2 x1", "x1 x2", "x2 d1", "x1 y1",
                 "y1 d1", "x1 o1"]:
        a, b = pair.split()
        L[a + b] = t.link(hubs[a], hubs[b], "R", *HOURLY)
    return t, L


def test_r0_clauses_one_by_one():
    t, L = clause_toy()
    got = present(net_of(t.tables()), 1)
    X, D_ = 3, 5
    cases = [
        # (destination, links, present?, clause that decides)
        (D_, ["n1x1", "x1n2"], False, "c_first != c_last"),
        (D_, ["n1x1", "x1d1"], True, "control"),
        (D_, ["ddx1"], False, "c_first != d"),
        (X, ["ddx1"], True, "control"),
        (D_, ["n1o2"], False, "c_last != o"),
        (X, ["n1x1"], True, "control"),
        (X, ["n1o1", "o1x1"], False, "transfer city is o"),
        (D_, ["o1d1", "d1x1"], False, "transfer city is d"),
        (X, ["o1d1", "d1x1"], True, "control"),
        (X, ["n1n2", "n2x1"], False, "transfer city is c_first"),
        (D_, ["o1x1", "x1x2"], False, "transfer city is c_last"),
        (D_, ["o1x1", "x1d1"], True, "control"),
        (D_, ["o1x1", "x1x2", "x2d1"], False, "transfers 1 and 2 share a city"),
        (D_, ["o1x1", "x1y1", "y1d1"], True, "control"),
        (D_, ["o1x1", "x1o1", "o1d1"], False, "loop (hub o1 twice)"),
    ]
    for d, names, want, clause in cases:
        assert ((d, *[L[k] for k in names]) in got) == want, clause
    # every candidate also passes the reference
    tb = t.tables()
    assert got == {(k[0], *[x for x in k[1:] if x >= 0]) for k in brute(tb, 1)}


def test_t_max_and_max_transfers():
    t, L = clause_toy()
    net = net_of(t.tables())
    c = service(SR.candidates(net, 1, [5], P))
    T = {(int(r.l1), int(r.l2), int(r.l3)): float(r.T) for r in c.itertuples()}
    direct, one = T[(L["o1d1"], -1, -1)], T[(L["o1x1"], L["x1d1"], -1)]
    assert direct < one
    for t_max, want in ((one, True), (one - 0.5, False)):
        got = present(net, 1, params(T_MAX_MIN=t_max))
        assert (5, L["o1d1"]) in got
        assert ((5, L["o1x1"], L["x1d1"]) in got) == want
    assert max(len(k) - 1 for k in present(net, 1)) == 3
    assert max(len(k) - 1 for k in present(net, 1, params(MAX_TRANSFERS=1))) == 2


def word_toy():
    """Paths R-A-R, A-R-R and A-R-A from O = 1 to D = 5 through airports
    aO (city 1), aX (3), aY (4), aD (5) and moves of 10 min to stations."""
    t = Toy()
    o1, x1, y1, d1 = t.hub(1), t.hub(3), t.hub(4), t.hub(5)
    aO, aX, aY, aD = (t.hub(iata=c) for c in ("OOO", "XXX", "YYY", "DDD"))
    for fua, h in ((1, o1), (1, aO), (3, aX), (4, aY), (5, d1), (5, aD)):
        t.connect(fua, h, 20)
    t.move(x1, aX, 10)
    t.move(y1, aY, 10)
    L = dict(
        o1x1=t.link(o1, x1, "R", (hm(6), 60)),
        aXaY=t.link(aX, aY, "A", (hm(9), 90)),
        y1d1=t.link(y1, d1, "R", (hm(12), 60)),
        aOaX=t.link(aO, aX, "A", (hm(7), 60)),
        x1y1=t.link(x1, y1, "R", (hm(10), 60)),
        aYaD=t.link(aY, aD, "A", (hm(13), 60)))
    return t, L


def test_mode_words():
    t, L = word_toy()
    tb = t.tables()
    net = net_of(tb)
    rar = (5, L["o1x1"], L["aXaY"], L["y1d1"])
    arr = (5, L["aOaX"], L["x1y1"], L["y1d1"])
    ara = (5, L["aOaX"], L["x1y1"], L["aYaD"])
    for rule, want in (("air_block_max2", (True, True)), ("deck35", (False, True)),
                       ("oct21", (True, False))):
        got = present(net, 1, params(MODE_WORDS=rule))
        assert (rar in got, arr in got) == want, rule
        assert ara not in got
    # the chain A-R-A exists: the grammar alone cuts it
    ap, pa = WINDOW["AP"], WINDOW["PA"]
    res = TR.triple_stats(tb["inst"], tb["typ"], tb["cal"], tb["trip"], *ara[1:],
                          10.0, ap[0], ap[1], 10.0, pa[0], pa[1],
                          P["MIN_COMMON_DATES"], True, TR.workspace(tb["inst"], tb["typ"]))
    assert chr(res[0]) == "T" and res[1] == 7
    c = SR.candidates(net, 1, [5], P)
    assert set(c.loc[c["seq"].isin(["RAR", "ARR"]), "basis"]) == {"T"}
    for rule, n in (("air_block_max2", 36), ("deck35", 35), ("oct21", 28)):
        allowed = SR.word_tables(params(MODE_WORDS=rule))[0]
        assert allowed.sum() == n
        assert [w for w, ok in zip(SR.WORDS, allowed) if ok] == [
            w for w in SR.WORDS if ref_word_ok(w, rule)]


def test_existence_thresholds():
    """Two links and three links on basis 'D' with opportunities on 4 and 5
    dates (MIN_ACTIVE_DATES = 5); rail to air on typical days with an
    opportunity on one weekday."""
    assert P["MIN_ACTIVE_DATES"] == 5
    t = Toy()
    o1, x1, y1, d1, aX, aD = t.hub(1), t.hub(3), t.hub(4), t.hub(5), t.hub(iata="XXX"), t.hub(iata="DDD")
    t.connect(1, o1, 10)
    t.connect(5, d1, 10)
    t.connect(3, aX, 30)
    t.connect(5, aD, 30)
    t.move(x1, aX, 10)
    a = t.link(o1, x1, "R", (hm(8), 60))
    two = {k: t.link(x1, d1, "R", (hm(9, 10) + k, 60, DAYS[:k]), (hm(15), 60 + k))
           for k in (4, 5)}
    mid = {k: t.link(x1, y1, "R", (hm(9, 10) + k, 60, DAYS[:k]), (hm(15), 60 + k))
           for k in (4, 5)}
    c = t.link(y1, d1, "R", (hm(10, 40), 60))
    air = t.link(aX, aD, "A", (hm(11), 60, MONDAYS))
    tb = t.tables()
    net = net_of(tb)
    rows = tb["transfers"].set_index(["link_in", "link_out"])
    for k in (4, 5):
        assert tuple(rows.loc[(a, two[k]), ["basis", "n_days"]]) == ("D", k)
    got = present(net, 1)
    assert (5, a, two[4]) not in got and (5, a, two[5]) in got
    assert (5, a, mid[4], c) not in got and (5, a, mid[5], c) in got
    cands = SR.candidates(net, 1, [5], P).set_index(["l1", "l2", "l3"])
    assert tuple(cands.loc[(a, mid[5], c), ["basis", "n_days"]]) == ("D", 5)
    assert tuple(cands.loc[(a, air, -1), ["basis", "n_days"]]) == ("T", 1)
    # the same opportunity counted by date is too rare
    tb["transfers"].loc[(tb["transfers"]["link_in"] == a)
                        & (tb["transfers"]["link_out"] == air), "basis"] = "D"
    assert (5, a, air) not in present(net_of(tb), 1)


# --- 2, 3. Values and pass-1 minima against the brute force --------------------
def test_candidates_equal_brute_force(rand):
    """Every R0 candidate with its values: T, gtt, ... as cost.path_costs gives
    them, three-link transfer times, frequency, basis and days as
    transfer.triple_stats gives them."""
    n3 = basis_t = 0
    for tb, net in rand:
        for o in (10, 11, 12, 13):
            c = SR.candidates(net, o, None, P)
            same_rows(as_dict(service(c)), brute(tb, o))
            n3 += int((c["n"] == 2).sum())
            basis_t += int((c["basis"] == "T").sum())
            car = c[c["seq"] == "C"]
            minutes = tb["car"].set_index(["o", "d"])["minutes"]
            m = minutes.loc[[(o, int(d)) for d in car["d"]]].to_numpy(np.float64)
            assert len(car) and (car["T"].to_numpy() == m).all()
            assert (car["gtc"].to_numpy() == (P["BETA_IVT"]["C"] * m).astype(np.float32)).all()
            assert (car["hwt"] == 0).all() and np.isinf(car["freq"]).all()
    assert n3 >= 30 and basis_t >= 30


def test_minima_and_best_by_sequence(rand):
    for tb, net in rand:
        for o in (10, 11, 12):
            ref = brute(tb, o)
            best, gtc, low_T = {}, {}, {}
            for (d, *chain), v in ref.items():
                g = np.float32(v["gtt"])
                key = (d, v["seq"])
                if key not in best or (g, tuple(chain)) < best[key]:
                    best[key] = (g, tuple(chain))
                cls = (d, "".join(sorted(set(v["seq"]))))
                gtc[cls] = min(gtc.get(cls, np.inf), np.float32(v["gtc"]))
                low_T[d] = min(low_T.get(d, np.inf), np.float32(v["T"]))
            per_word, per_class = SR.minima(net, o, P)
            assert {(r.d, r.seq): (np.float32(r.min_gtt), (r.l1, r.l2, r.l3))
                    for r in per_word.itertuples()} == best
            assert {(r.d, r.cls): np.float32(r.min_gtc) for r in per_class.itertuples()} == gtc
            out = SR.run_origin(net, o, P)
            b = service(out["best"])
            assert {(r.d, r.seq): (r.gtt, (r.l1, r.l2, r.l3)) for r in b.itertuples()} == best
            same_rows(as_dict(b), {(d, *ch): ref[(d, *ch)] for (d, _), (_, ch) in best.items()})
            f = out["funnel"].set_index("d")
            for d, t_min in low_T.items():
                assert f.loc[d, "best_T"] == t_min
                assert f.loc[d, "best_gtt"] == min(g for (dd, _), (g, _) in best.items() if dd == d)


# --- 4. R3 in the generation ---------------------------------------------------
@pytest.mark.parametrize("mu", [P["MU"], 0.1, 0.0])
def test_r3_in_generation_equals_r3_after(rand, mu):
    cut_somewhere = False
    for tb, net in rand:
        for o in (10, 11, 12):
            full = SR.candidates(net, o, None, P, mu=math.inf)
            cls = full["seq"].map(lambda s: "".join(sorted(set(s))))
            g, h = full["gtt"].astype(float), full["gtc"].astype(float)
            keep = ((g <= (1 + mu) * g.groupby([full["d"], cls]).transform("min"))
                    | (h <= (1 + mu) * h.groupby([full["d"], cls]).transform("min")))
            cut_somewhere |= not keep.all()
            pd.testing.assert_frame_equal(full[keep].reset_index(drop=True),
                                          SR.candidates(net, o, None, P, mu=mu))
            if mu == P["MU"]:
                f = SR.run_origin(net, o, P)["funnel"].set_index("d")
                kept = keep.groupby(full["d"]).sum()
                assert (f.loc[kept.index, "n_r3"] == kept).all()
                r0 = full.groupby("d").size()
                assert (f.loc[r0.index, "n_r0"] >= r0).all()
    assert cut_somewhere


# --- 5. Count from the arc degrees ---------------------------------------------
def layered_toy(seed):
    """O = 1 with one hub, then three layers of hubs, each its own FUA, links
    only forward with 0..2 parallel links (riding-time classes) per hub pair,
    every transfer feasible daily; two hubs also serve FUA 99. No city rule
    can bite."""
    rng = np.random.default_rng(seed)
    t = Toy()
    h0 = t.hub(1)
    t.connect(1, h0, 10)
    layers = [[t.hub(10 * (k + 1) + i) for i in range(n)] for k, n in enumerate((3, 3, 2))]
    for k, layer in enumerate(layers):
        for i, h in enumerate(layer):
            t.connect(10 * (k + 1) + i, h, 15)
    t.connect(99, layers[1][0], 30)
    t.connect(99, layers[2][1], 30)
    for k, (from_, to) in enumerate(zip([[h0]] + layers[:-1], layers)):
        for a in from_:
            for b in to:
                for c in range(int(rng.integers(0 if k else 1, 3))):
                    t.link(a, b, "R", (hm(6) + 90 * k, 50 + 5 * c))
    return t, h0


@pytest.mark.parametrize("seed", [4, 5])
def test_count_from_arc_degrees(seed):
    t, h0 = layered_toy(seed)
    tb = t.tables()
    net = net_of(tb)
    n_hubs = len(t.hubs)
    arcs = np.zeros((n_hubs, n_hubs))
    for src, trg, _, _ in t.links:
        arcs[src, trg] += 1
    ends = np.zeros(n_hubs)
    for fua, hub, _, _ in t.conn:
        ends[hub] += fua != 1
    walk = np.eye(n_hubs)[h0]
    want = 0
    for _ in range(P["MAX_TRANSFERS"] + 1):
        walk = walk @ arcs
        want += walk @ ends
    assert want > 20
    c = SR.candidates(net, 1, None, P)
    assert len(service(c)) == want
    funnel = SR.run_origin(net, 1, P)["funnel"]
    assert funnel["n_r0"].sum() == want


# --- 6. Bounds -----------------------------------------------------------------
def test_bounds_hold_on_every_candidate(rand):
    par = C.param_array(P)
    for tb, net in rand:
        links = tb["links"]
        code = links["mode"].map(C.MODE_CODE).to_numpy()
        for o in (10, 11, 12):
            c = service(SR.candidates(net, o, None, P))
            n = c["n"].to_numpy(np.int64) + 1
            ls = c[["l1", "l2", "l3"]].to_numpy()
            used = ls >= 0
            modes = np.where(used, code[ls], 0)
            rides = np.where(used, links["ride"].to_numpy()[ls], 0.0)
            fmax = np.where(used, links["freq_max"].to_numpy()[ls], 0.0)
            lo = np.array([[WINDOW[S.window_key(S.MODES[m[k]], S.MODES[m[k + 1]])][0]
                            for k in range(2)] for m in modes])
            moves = c[["mv1", "mv2"]].fillna(0).to_numpy()
            low = C.path_bounds(n, c["acc"], modes, rides, lo + moves, c["egr"], fmax, par)
            fts = c[["ft1", "ft2"]].fillna(0).to_numpy()
            true = C.path_costs(n, c["acc"], modes, rides, fts, c["egr"], c["freq"], par)
            assert C.bounds_hold(low, true).all()
            assert (c["freq"] <= c["freq_counted"]).all()
            assert (c["freq"] <= fmax.min(axis=1, initial=np.inf, where=used)).all()


@pytest.mark.parametrize("shift", [1e-6, 1e3])
def test_violated_bound_stops_the_shard(rand, monkeypatch, shift):
    _, net = rand[0]
    monkeypatch.setattr(SR, "BOUND_TEST_SHIFT", shift)
    with pytest.raises(SR.BoundViolation):
        SR.run_origin(net, 10, P)
    with pytest.raises(SR.BoundViolation):
        list(SR.run_pool([10, 11], P, 1, net=net))


# --- 7. Determinism ------------------------------------------------------------
def shuffled(tb: dict, seed: int) -> dict:
    out = dict(tb)
    for k in ("hubs", "links", "transfers", "connectors", "car"):
        out[k] = tb[k].sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def test_output_independent_of_order_workers_and_blocks(rand):
    tb, net = rand[0]
    origins = [10, 11, 12, 13]
    prm = params(NAMED_PAIRS=[[10, 13], [12, 15], [11, 99]])

    def run(order, workers, network, p):
        return dict(SR.run_pool(order, p, workers, named=True, net=network))

    base = run(origins, 1, net, prm)
    others = [run(origins[::-1], 2, net_of(shuffled(tb, 7)), prm),
              run(origins, 1, net, dict(prm, MEM_CAND_GB=1e-12))]
    assert base[10]["funnel"].attrs["search"]["blocks"] == 1
    assert others[1][10]["funnel"].attrs["search"]["blocks"] > 2
    assert len(base[10]["trace"]) and len(base[12]["trace"]) and not len(base[11]["trace"])
    for other in others:
        for o in origins:
            for table, df in base[o].items():
                got = other[o][table].copy()
                got.attrs = {}
                want = df.copy()
                want.attrs = {}
                pd.testing.assert_frame_equal(got, want, check_exact=True)


# --- 8. Contract ---------------------------------------------------------------
F32 = "float32"
PATH_TYPES = {"o": "int16", "d": "int16", "seq": "str", "n": "int8", "h_first": "int32",
              "h_last": "int32", "l1": "int32", "l2": "int32", "l3": "int32",
              "cities": "str", "key": "str",
              **{c: F32 for c in ("acc", "egr", "mv1", "mv2", "ft1", "ft2", "t_a", "t_dep",
                                  "t_ivt", "t_trans", "t_tp", "t_arr", "t_e", "T", "gtt",
                                  "hwt", "hwt_first", "gtc", "freq", "freq_counted")},
              "basis": "str", "n_days": "int32"}
CONTRACT = {
    "alternatives": {**PATH_TYPES, "rank": "int16", "cls": "str", "freq_pooled": F32,
                     "n_variants": "int32", "status": "str"},
    "variants": {"o": "int16", "d": "int16", "rank": "int16", "member": "int16",
                 **{k: v for k, v in PATH_TYPES.items() if k not in ("o", "d")}},
    "best": PATH_TYPES,
    "funnel": {"o": "int16", "d": "int16",
               **{k: "int64" for k in ("n_r0", "n_r3", "n_r1", "n_r2", "n_kept", "n_capped")},
               "n_classes": "int8", "best_T": F32, "best_gtt": F32, "car_min": F32, "pt": "bool"},
    "seq_counts": {"o": "int16", "seq": "str", "n_r0": "int64", "n_r3": "int64",
                   "n_kept": "int64"},
    "unserved": {"o": "int16", "d": "int16", "reason": "str"},
    "trace": {**PATH_TYPES, "cand": "int64", "cls": "str", "alt": "int64", "status": "str",
              "rule": "str", "by": "int64", "detail": "str"},
}


def check_contract(df: pd.DataFrame, table: str) -> None:
    want = CONTRACT[table]
    assert list(df.columns) == list(want), table
    for col, t in want.items():
        if t == "str":
            assert pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object, (table, col)
        else:
            assert df[col].dtype == np.dtype(t), (table, col, df[col].dtype)


def test_shard_tables_follow_the_contract(rand):
    tb, net = rand[1]
    out = SR.run_origin(net, 10, P, named_dests=[13, 14])
    assert set(out) == {"alternatives", "variants", "best", "funnel", "seq_counts", "trace"}
    for table, df in out.items():
        check_contract(df, table)
    a, v, f, t = out["alternatives"], out["variants"], out["funnel"], out["trace"]
    assert len(a) and len(v) and len(t)
    for d, grp in a.groupby("d"):
        assert grp["rank"].tolist() == list(range(1, len(grp) + 1))
        assert (np.diff(grp["gtc"].to_numpy()) >= 0).all()
    assert set(a["status"]) <= {"kept", "kept_guarantee"}
    assert (a.groupby("d").size() == f.set_index("d").loc[a["d"].unique(), "n_kept"]).all()
    assert v["member"].max() <= P["VARIANTS_KEPT"]
    assert (f["n_r0"] >= f["n_r3"]).all() and (f["n_r3"] >= f["n_r1"]).all()
    assert (f["n_r1"] >= f["n_r2"]).all() and (f["n_r2"] == f["n_kept"] + f["n_capped"]).all()
    assert set(f["d"]) == set(net.fuas) - {10}
    sc = out["seq_counts"]
    assert sc["n_r0"].sum() == f["n_r0"].sum() and sc["n_kept"].sum() == len(a)
    assert set(t["d"]) == {13, 14} and t["status"].str.len().min() > 0
    exact = SR.candidates(net, 10, [13, 14], P).groupby("d").size()
    assert (t.groupby("d").size() == exact).all()
    assert (exact <= f.set_index("d").loc[[13, 14], "n_r0"]).all()
    # the alternatives of a traced pair are those of the bounded run
    kept = t[t["status"].isin(["kept", "kept_guarantee"])].sort_values(["d", "gtc", "key"])
    alt = a[a["d"].isin([13, 14])].sort_values(["d", "gtc", "key"])
    assert kept["key"].tolist() == alt["key"].tolist()


def load_script():
    spec = importlib.util.spec_from_file_location("pathset_enumerate",
                                                  REPO / "pathset-enumerate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_shards_combine_and_resume(rand, tmp_path, monkeypatch):
    """pathset-enumerate.py end to end on a synthetic PATHSET_DIR (tmp_path):
    shards, combined tables, ps-unserved, the manifest and --resume."""
    tb, _ = rand[0]
    monkeypatch.setattr(io, "OUT", tmp_path.resolve())
    monkeypatch.setitem(io.P, "NAMED_PAIRS", [[10, 13]])
    for name in ("hubs", "links", "events", "transfers", "connectors", "car"):
        io.save(tb[name], f"ps-{name}")
    io.save_npy(tb["masks"], "ps-event-masks")
    script = load_script()
    info = script.run(script.parse_args(["--origins", "all", "--workers", "1",
                                         "--named", "--combine"]))
    fuas = list(range(10, 16))
    assert info["origins_run"] == fuas
    for table, (name, _) in script.COMBINED.items():
        for o in fuas:
            assert (tmp_path / "paths" / table / f"o{o:03d}.parquet").exists()
        df = io.load_out(name)
        check_contract(df, table)
    funnel = io.load_out("ps-funnel")
    assert len(funnel) == len(fuas) * (len(fuas) - 1)
    un = io.load_out("ps-unserved")
    check_contract(un, "unserved")
    lost = funnel[~funnel["pt"]]
    assert un[["o", "d"]].values.tolist() == lost[["o", "d"]].values.tolist()
    assert (un["reason"] == np.where(lost["car_min"].isna(), "nothing", "no_service_path")).all()
    assert set(io.load_out("ps-trace")["d"]) == {13}
    assert (tmp_path / "ps-manifest.jsonl").exists()
    # resume: nothing to do with the same hash, everything with another MU
    again = script.run(script.parse_args(["--origins", "all", "--workers", "1",
                                          "--named", "--resume"]))
    assert again["origins_run"] == [] and again["origins_skipped"] == fuas
    other = script.run(script.parse_args(["--origins", "10,11", "--workers", "1",
                                          "--named", "--resume", "--mu", "inf"]))
    assert other["origins_run"] == [10, 11]
    with pytest.raises(RuntimeError):
        script.combine()                       # shards of two runs
    assert script.select_origins("stagec", dict(io.P, STAGE_C_ORIGINS=[10, 12])) == [10, 12]
    with pytest.raises(ValueError):
        script.select_origins("10,77", io.P)


def script_inputs(tb: dict, tmp_path, monkeypatch):
    """The input tables of ``tb`` in a synthetic PATHSET_DIR (tmp_path)."""
    monkeypatch.setattr(io, "OUT", tmp_path.resolve())
    for name in ("hubs", "links", "events", "transfers", "connectors", "car"):
        io.save(tb[name], f"ps-{name}")
    io.save_npy(tb["masks"], "ps-event-masks")
    return load_script()


def test_interrupted_save_is_neither_resumed_nor_combined(rand, tmp_path, monkeypatch):
    """Every shard carries the run hash: an origin with a shard of another run
    (a save interrupted before its funnel) or a missing shard is rerun by
    --resume and refused by --combine."""
    script = script_inputs(rand[0][0], tmp_path, monkeypatch)
    args = ["--origins", "10,11", "--workers", "1"]
    script.run(script.parse_args(args))
    h = script.shard_hash(10)
    assert h is not None and script.shard_hash(11) == h
    for table in script.COMBINED:
        assert script.file_hash(tmp_path / "paths" / table / "o010.parquet") == h
    # another run wrote the variants of origin 11, then stopped
    v = io.load_out(script.shard_name("variants", 11))
    v.attrs["param_hash"] = "another-run"
    io.save(v, script.shard_name("variants", 11))
    assert script.shard_hash(11) is None
    with pytest.raises(RuntimeError, match="origin 11"):
        script.combine()
    again = script.run(script.parse_args(args + ["--resume"]))
    assert again["origins_run"] == [11] and again["origins_skipped"] == [10]
    assert script.combine()["ps-funnel"] == 2 * (len(rand[0][1].fuas) - 1)
    # a shard without the others
    (tmp_path / "paths" / "funnel" / "o011.parquet").unlink()
    with pytest.raises(RuntimeError, match="origin 11"):
        script.combine()


def test_windows_must_be_those_of_the_network(rand):
    _, net = rand[0]
    windows = copy.deepcopy(P["WINDOWS"])
    windows["PP"] = [windows["PP"][0] + 1, windows["PP"][1]]
    p = params(WINDOWS=windows)
    with pytest.raises(ValueError, match="WINDOWS"):
        SR.run_origin(net, 10, p)
    with pytest.raises(ValueError, match="WINDOWS"):
        SR.candidates(net, 10, None, p)


# --- Memory --------------------------------------------------------------------
def strip(out: dict) -> dict:
    for df in out.values():
        df.attrs = {}
    return out


def test_cache_is_bounded_and_emptying_it_changes_nothing(rand, monkeypatch):
    """The triple_stats cache stays within MEM_CAND_GB (CACHE_BYTES an entry)
    and is emptied after pass 2; a cache emptied at every call, a buffer of
    many small chunks and one destination per block give the same tables, and
    no chain is walked more than once per pass whatever the blocks."""
    _, net = rand[0]
    room = 10
    small = params(MEM_CAND_GB=room * SR.CACHE_BYTES / 2 ** 30)
    sh = SR.Shard(net, 10, small)
    sh.minima(sh.want())
    rec = sh.log["pass1"]
    assert rec["triple_calls"] > 3 * room and rec["cache_cleared"] > 0
    assert 0 < len(sh.cache) <= room
    full = SR.Shard(net, 10, P)
    full.minima(full.want())
    assert full.log["pass1"]["cache_cleared"] == 0
    assert len(full.cache) == full.log["pass1"]["triple_calls"]

    for o in (10, 12):
        base = SR.run_origin(net, o, P, named_dests=[13, 15])
        log = base["funnel"].attrs["search"]
        assert set(k for k in log if k.startswith(("pass", "trace"))) == \
            {"pass1", "pass2", "trace_pass2"}
        assert log["blocks"] == 1
        monkeypatch.setattr(SR, "CHUNK", 3)
        tiny = SR.run_origin(net, o, params(MEM_CAND_GB=1e-12), named_dests=[13, 15])
        monkeypatch.undo()
        t = tiny["funnel"].attrs["search"]
        assert t["blocks"] > 2 and t["pass2"]["cache_cleared"] > 0
        assert t["pass1"]["cache_entries"] <= 1 and t["pass2"]["cache_entries"] <= 1
        for k in ("chains1", "chains2", "chains3", "evaluated", "written"):
            assert t["pass2"][k] == log["pass2"][k]
        strip(base)
        strip(tiny)
        for table, df in base.items():
            pd.testing.assert_frame_equal(tiny[table], df, check_exact=True)


def test_candidates_from_a_buffer_of_small_chunks(rand, monkeypatch):
    _, net = rand[1]
    want = {mu: SR.candidates(net, 11, None, P, mu=mu) for mu in (math.inf, P["MU"])}
    monkeypatch.setattr(SR, "CHUNK", 2)
    for mu, df in want.items():
        got = SR.candidates(net, 11, None, params(MEM_CAND_GB=1e-12), mu=mu)
        pd.testing.assert_frame_equal(got, df, check_exact=True)
    sh = SR.Shard(net, 11, P)
    sh.collect(sh.want())
    assert sh.n_ids > 10 and len(sh.st.buf) == -(-sh.n_ids // 2)
    assert len(sh.kept_ids(sh.want())) == sh.n_ids == sh.st.n_r3.sum()


def test_r0_counted_by_bounds_is_logged(rand):
    """n_r0 is exact at MU = inf; at a finite MU the part counted by bounds is
    in the log, and the exact count lies between n_r0 less that part and
    n_r0 (module docstring, "Counting convention")."""
    tb, net = rand[0]
    counted = 0
    for o in (10, 11, 12):
        exact = len(service(SR.candidates(net, o, None, P)))
        out = SR.run_origin(net, o, P)
        sc = out["seq_counts"]
        n_r0 = int(sc.loc[sc["seq"] != "C", "n_r0"].sum())
        by_bounds = out["funnel"].attrs["search"]["pass2"]["r0_by_bounds"]
        counted += by_bounds
        assert n_r0 - by_bounds <= exact <= n_r0
        inf = SR.run_origin(net, o, params(MU=math.inf))
        sc = inf["seq_counts"]
        assert inf["funnel"].attrs["search"]["pass2"]["r0_by_bounds"] == 0
        assert int(sc.loc[sc["seq"] != "C", "n_r0"].sum()) == exact
    assert counted > 0


# --- Smaller rules -------------------------------------------------------------
def test_airport_city_and_destination_blocks():
    hubs = pd.DataFrame({"hub": [0, 1, 2, 3], "kind": ["S", "A", "A", "A"],
                         "fua": [7, -1, -1, -1]})
    conn = pd.DataFrame({"fua": [9, 8, 5, 6], "hub": [1, 1, 2, 2],
                         "acc_min": [30.0, 30.0, np.nan, 40.0]})
    assert SR.hub_cities(hubs, conn).tolist() == [7, 8, 6, -4]
    kept = np.array([0, 5, 1, 0, 3])
    want = np.array([False, True, True, True, True])
    per = SR.CAND_BYTES
    assert SR.dest_blocks(kept, want, 6 * per) == [[1, 2, 3], [4]]
    assert SR.dest_blocks(kept, want, 0) == [[1], [2], [3], [4]]
    assert SR.dest_blocks(kept, want, np.inf) == [[1, 2, 3, 4]]
