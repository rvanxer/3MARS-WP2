"""Reviewer's checks of pathset/search.py, independent of tests/test_search.py.

A second synthetic generator (stations and airports in several time zones,
station moves, 0-minute colocated airport-station moves and road moves,
departures shortly before midnight, thin and weekday calendars, duplicated
links for ties, shared trip keys, an airport without a city, a FUA served only
by an airport, a car-only FUA) and a plain-Python reference that shares no
code with search.py or with the brute force of test_search.py. The reference
takes values from pathset.cost, three-link statistics from
pathset.transfer.triple_stats and the rules from pathset.rules.apply_rules.
"""

import copy
import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pathset import cost as C  # noqa: E402
from pathset import dates as D  # noqa: E402
from pathset import rules as RU  # noqa: E402
from pathset import schemas as S  # noqa: E402
from pathset import search as SR  # noqa: E402
from pathset import transfer as TR  # noqa: E402
from pathset.io import P  # noqa: E402

DAYS = tuple(range(200, 240))


def prm(**kw):
    out = copy.deepcopy(P)
    out.update(kw)
    return out


# --- Generator -----------------------------------------------------------------
def generate(seed, p, n_links=110):
    """Hubs (kind, fua, iata, tz), links (src, trg, mode, events), connectors
    {(fua, hub): (acc, egr)}, moves {(a, b): minutes}, car rows."""
    rng = np.random.default_rng(seed)
    fuas = list(range(20, 20 + int(rng.integers(4, 7))))
    tz = {f: int(rng.integers(0, 3)) for f in fuas}
    hubs, st_of, moves, conn = [], {}, {}, {}
    for f in fuas:
        st_of[f] = []
        for _ in range(1 if rng.random() < 0.5 else 2):
            hubs.append(("S", f, "", tz[f]))
            st_of[f].append(len(hubs) - 1)
        if len(st_of[f]) == 2:
            m = float(rng.integers(0, 13))
            moves[tuple(st_of[f])] = moves[tuple(st_of[f][::-1])] = m
        for h in st_of[f]:
            conn[(f, h)] = (float(rng.integers(5, 31)), float(rng.integers(5, 31)))
    stations = [h for f in fuas for h in st_of[f]]
    for f, g in zip(fuas, fuas[1:]):
        if rng.random() < 0.4:
            conn[(g, st_of[f][0])] = (float(rng.integers(40, 61)), float(rng.integers(40, 61)))
    airports = []
    for i in range(int(rng.integers(3, 6))):
        hubs.append(("A", -1, f"A{i}{chr(65 + seed % 26)}", int(rng.integers(0, 3))))
        airports.append(len(hubs) - 1)
    for j, a in enumerate(airports[:-1]):            # the last airport: no city
        cat = rng.choice(fuas, size=int(rng.integers(1, 4)), replace=False)
        accs = [float(rng.integers(20, 111)) for _ in cat]
        if j == 0 and len(cat) >= 2:
            accs[1] = accs[0]                         # tie of access times
        for f, acc in zip(cat, accs):
            conn[(int(f), a)] = (acc, float(rng.integers(20, 111)))
        if j == 1:
            conn[(80, a)] = (float(rng.integers(30, 90)), float(rng.integers(30, 90)))
    for a in airports:
        if rng.random() < 0.6:
            s = int(rng.choice(stations))
            m = 0.0 if rng.random() < 0.5 else float(rng.integers(10, 41))
            moves[(a, s)] = m
            moves[(s, a)] = m if rng.random() < 0.7 else m + 5
    pool = [DAYS, DAYS, tuple(d for d in DAYS if D.index_date(d).weekday() < 5),
            tuple(d for d in DAYS if D.index_date(d).weekday() >= 5), DAYS[:6], DAYS[10:15],
            DAYS[::7], tuple(range(200, 212)), tuple(range(226, 240))]
    trip = itertools.count(1)
    links = []
    for _ in range(int(rng.integers(n_links // 2, n_links + 1))):
        a, b = rng.choice(len(stations), 2, replace=False)
        days = pool[int(rng.integers(0, len(pool)))]
        ride, start = int(rng.integers(4, 60)) * 5, int(rng.integers(0, 288)) * 5
        head = int(rng.choice([20, 30, 60, 120]))
        evs = []
        for k in range(int(rng.integers(2, 10))):
            dep = (start + k * head) % 1440 if rng.random() < 0.8 else int(rng.integers(1350, 1440))
            evs.append((dep, ride + int(rng.choice([0, 0, 5, 10])), next(trip),
                        days if rng.random() < 0.8 else pool[int(rng.integers(0, len(pool)))]))
        links.append([int(stations[a]), int(stations[b]), "RB"[int(rng.integers(0, 2))], evs])
    for _ in range(int(rng.integers(2 * len(airports), 5 * len(airports)))):
        a, b = rng.choice(len(airports), 2, replace=False)
        days = pool[int(rng.integers(0, len(pool)))]
        links.append([airports[a], airports[b], "A",
                      [(int(rng.integers(0, 288)) * 5, int(rng.integers(10, 40)) * 5, next(trip), days)
                       for _ in range(int(rng.integers(1, 5)))]])
    for _ in range(2):                                 # duplicated links: ties
        k = int(rng.integers(0, len(links)))
        links.append([*links[k][:3], [(e[0], e[1], next(trip), e[3]) for e in links[k][3]]])
    by_src = defaultdict(list)
    for i, l in enumerate(links):
        by_src[l[0]].append(i)
    for i, l in enumerate(links):                     # a vehicle that continues
        if rng.random() < 0.15 and by_src[l[1]]:
            j = int(rng.choice(by_src[l[1]]))
            if j != i:
                e0, f0 = l[3][0], links[j][3][0]
                links[j][3][0] = (min(1439, (e0[0] + e0[1] + 10) % 1440), f0[1], e0[2], f0[3])
    links = [l for l in links
             if len(set().union(*[set(e[3]) for e in l[3]])) >= p["MIN_ACTIVE_DATES"]]
    car = [(o, d, float(rng.integers(60, 600))) for o in fuas + [80, 90]
           for d in fuas + [80, 90] if o != d and rng.random() < 0.6]
    return hubs, links, conn, moves, car


def contract_tables(hubs, links, conn, moves, car, p) -> dict:
    H = len(hubs)
    th = pd.DataFrame({
        "hub": np.arange(H, dtype=np.int32), "kind": [h[0] for h in hubs],
        "name": [f"h{i}" for i in range(H)], "fua": np.array([h[1] for h in hubs], np.int16),
        "icc": "XX", "tz_std": np.array([h[3] for h in hubs], np.int8), "lon": 0.0, "lat": 0.0,
        "iata": [h[2] for h in hubs], "stns": [[] for _ in range(H)], "released_id": "",
        "match_m": np.float32(np.nan)})
    ev_rows, mask_rows, rows = [], [], []
    for l, (src, trg, mode, evs) in enumerate(links):
        evs = sorted(evs)
        ev0 = len(ev_rows)
        cnt = np.zeros(D.N_DAYS, dtype=np.int64)
        for dep, ride, trip, days in evs:
            ev_rows.append((l, dep, ride, trip, len(set(days))))
            r = np.zeros(D.N_DAYS, dtype=bool)
            r[list(days)] = True
            mask_rows.append(r)
            cnt += r
        pos = cnt[cnt > 0]
        rides = np.array([e[1] for e in evs], float)
        w = np.array([len(set(e[3])) for e in evs], float)
        rows.append(dict(link=l, src=src, trg=trg, mode=mode, cls=0,
                         ride=(rides * w).sum() / w.sum(), ride_min=rides.min(),
                         ride_max=rides.max(), tz_shift=60 * (hubs[trg][3] - hubs[src][3]),
                         km=1.0, ev0=ev0, ev1=len(ev_rows), n_events=len(evs),
                         active_dates=len(pos), first_day=int(np.flatnonzero(cnt)[0]),
                         last_day=int(np.flatnonzero(cnt)[-1]), freq=float(np.median(pos)),
                         freq_max=int(pos.max()), freq_span=0.0, freq_fixed=0.0,
                         active_share=1.0))
    tl = pd.DataFrame(rows)[S.LINKS].astype({
        "link": "int32", "src": "int32", "trg": "int32", "cls": "int16", "ride": "float32",
        "ride_min": "int16", "ride_max": "int16", "tz_shift": "int16", "km": "float32",
        "ev0": "int32", "ev1": "int32", "n_events": "int32", "active_dates": "int32",
        "first_day": "int32", "last_day": "int32", "freq": "float32", "freq_max": "int16",
        "freq_span": "float32", "freq_fixed": "float32", "active_share": "float32"})
    te = pd.DataFrame(ev_rows, columns=S.EVENTS).astype(
        {"link": "int32", "dep": "int16", "ride": "int16", "trip": "int32", "n_dates": "int32"})
    masks = D.pack_bool(np.array(mask_rows))
    args = (te["dep"].to_numpy(), te["ride"].to_numpy(), masks, tl["ev0"].to_numpy(),
            tl["ev1"].to_numpy(), tl["tz_shift"].to_numpy())
    inst, typ = TR.build_instances(*args), TR.build_typical(*args)
    cal = TR.link_calendars(masks, tl["ev0"].to_numpy(np.int32), tl["ev1"].to_numpy(np.int32))
    trip = te["trip"].to_numpy(np.int32)
    by_src = defaultdict(list)
    for l, s in enumerate(tl["src"]):
        by_src[int(s)].append(l)
    nb = defaultdict(list)
    for (x, y), m in moves.items():
        nb[x].append((y, m))
    md = tl["mode"].tolist()
    pairs = [(li, lo, mv, *p["WINDOWS"][S.window_key(md[li], md[lo])],
              (md[li] == "A") != (md[lo] == "A"))
             for li, t in enumerate(tl["trg"].tolist()) for hub, mv in [(t, 0.0)] + nb[t]
             for lo in by_src[hub] if lo != li]
    cols = list(zip(*pairs)) if pairs else [()] * 6
    out = TR.transfer_table(inst, typ, cal, trip, *[np.array(c) for c in cols[:5]],
                            typical=np.array(cols[5], dtype=bool))
    tt = pd.DataFrame({
        "link_in": np.array(cols[0], np.int32), "link_out": np.array(cols[1], np.int32),
        "move": np.array(cols[2], np.float32), "basis": TR.basis_labels(out["basis"]),
        "n_days": out["n_days"], "freq": out["freq"], "ft_mean": out["ft_mean"],
        "ft_min": out["ft_min"]})
    tt = tt[tt["n_days"] >= 1].reset_index(drop=True)
    tc = pd.DataFrame([(f, h, a, e) for (f, h), (a, e) in sorted(conn.items())],
                      columns=["fua", "hub", "acc_min", "egr_min"]).astype(
        {"fua": "int16", "hub": "int32", "acc_min": "float32", "egr_min": "float32"})
    tc["source"] = "fitted"
    tcar = pd.DataFrame(car, columns=["o", "d", "minutes"]).astype(
        {"o": "int16", "d": "int16", "minutes": "float32"})
    tcar["km"], tcar["fixed"] = np.float32(1), False
    return dict(hubs=th, links=tl, events=te, masks=masks, transfers=tt, connectors=tc,
                car=tcar, inst=inst, typ=typ, cal=cal, trip=trip)


def network(tb, p):
    return SR.build_network(tb["hubs"], tb["links"], tb["events"], tb["masks"],
                            tb["transfers"], tb["connectors"], tb["car"], p)


# --- Reference -----------------------------------------------------------------
def word_ok(word, rule, n_max):
    air = [i for i, m in enumerate(word) if m == "A"]
    ok = len(word) <= n_max and len(air) <= 2 and (len(air) < 2 or air[1] == air[0] + 1)
    if rule == "deck35":
        return ok and word != "RAR"
    if rule == "oct21":
        return ok and word not in ("ABB", "ABR", "ARB", "ARR", "BBA", "BRA", "RBA", "RRA")
    return ok


def city_of(tb):
    c, city = tb["connectors"], {}
    for h, k, f in zip(tb["hubs"]["hub"], tb["hubs"]["kind"], tb["hubs"]["fua"]):
        rows = [(float(a), int(ff)) for ff, hh, a in zip(c["fua"], c["hub"], c["acc_min"])
                if hh == h and np.isfinite(a)]
        city[int(h)] = int(f) if k == "S" else (min(rows)[1] if rows else -1 - int(h))
    return city


def reference(tb, o, p):
    """{(d, l1, l2, l3): values} of every R0 candidate of origin o."""
    L = tb["links"]
    src, trg, mode = L["src"].tolist(), L["trg"].tolist(), L["mode"].tolist()
    ride, freq, fmax = L["ride"].tolist(), L["freq"].tolist(), L["freq_max"].tolist()
    city = city_of(tb)
    lab = {int(h): (str(int(f)) if k == "S" else i) for h, k, f, i in zip(
        tb["hubs"]["hub"], tb["hubs"]["kind"], tb["hubs"]["fua"], tb["hubs"]["iata"])}
    c = tb["connectors"]
    acc = {int(h): float(a) for f, h, a in zip(c["fua"], c["hub"], c["acc_min"])
           if f == o and np.isfinite(a)}
    egr = defaultdict(list)
    for f, h, e in zip(c["fua"], c["hub"], c["egr_min"]):
        egr[int(h)].append((int(f), float(e)))
    rows, nxt = {}, defaultdict(list)
    for r in tb["transfers"].itertuples(index=False):
        rows[(r.link_in, r.link_out)] = r
        nxt[r.link_in].append(r.link_out)
    n_max, par = p["MAX_TRANSFERS"] + 1, C.param_array(p)
    work = TR.workspace(tb["inst"], tb["typ"])
    chains, stack = [], [[l] for l in range(len(src)) if src[l] in acc]
    while stack:
        ch = stack.pop()
        chains.append(ch)
        stack += [ch + [b] for b in nxt[ch[-1]]] if len(ch) < n_max else []
    out = {}
    for ch in chains:
        n, word = len(ch), "".join(mode[l] for l in ch)
        if not word_ok(word, p["MODE_WORDS"], n_max):
            continue
        if n == 1:
            basis, nd, counted, ft = "D", int(L["active_dates"][ch[0]]), freq[ch[0]], []
        elif n == 2:
            r = rows[tuple(ch)]
            basis, nd, counted, ft = r.basis, int(r.n_days), float(r.freq), [float(r.ft_mean)]
        else:
            m = [mode[l] for l in ch]
            w1, w2 = (p["WINDOWS"][S.window_key(m[k], m[k + 1])] for k in (0, 1))
            res = TR.triple_stats(
                tb["inst"], tb["typ"], tb["cal"], tb["trip"], *ch,
                float(rows[(ch[0], ch[1])].move), w1[0], w1[1], float(rows[(ch[1], ch[2])].move),
                w2[0], w2[1], p["MIN_COMMON_DATES"],
                any((m[k] == "A") != (m[k + 1] == "A") for k in (0, 1)), work)
            basis, nd, counted, ft = chr(res[0]), int(res[1]), float(res[2]), [res[3], res[5]]
        if not (nd >= p["MIN_ACTIVE_DATES"] if basis == "D" else nd >= 1):
            continue
        hs, tcs, mv = [src[ch[0]], trg[ch[0]]], [], []
        for k in range(1, n):
            a, b = trg[ch[k - 1]], src[ch[k]]
            mv.append(float(rows[(ch[k - 1], ch[k])].move))
            hs += ([b] if a != b else []) + [trg[ch[k]]]
            tcs.append({city[a], city[b]})
        cf, cl = city[hs[0]], city[hs[-1]]
        if (len(set(hs)) < len(hs) or cf == cl or cl == o or any(t & {o, cf, cl} for t in tcs)
                or (n == 3 and tcs[0] & tcs[1])):
            continue
        pad = ch + [-1] * (3 - n)
        capped = float(C.capped_frequency([n], [counted], [[fmax[l] if l >= 0 else 0 for l in pad]])[0])
        for d, e in egr[hs[-1]]:
            if not np.isfinite(e) or d == o or d == cf or any(d in t for t in tcs):
                continue
            cost = C.path_costs([n], [acc[hs[0]]], [[C.MODE_CODE[mode[l]] if l >= 0 else 0 for l in pad]],
                                [[ride[l] if l >= 0 else 0.0 for l in pad]], [(ft + [0.0, 0.0])[:2]],
                                [e], [capped], par)
            if not cost["T"][0] <= p["T_MAX_MIN"]:
                continue
            out[(d, *pad)] = dict(
                seq=word, n=n - 1, h_first=hs[0], h_last=hs[-1], key="-".join(map(str, ch)),
                cities=">".join(k for k, _ in itertools.groupby(lab[h] for h in hs)),
                acc=acc[hs[0]], egr=e, mv1=(mv + [np.nan] * 2)[0], mv2=(mv + [np.nan] * 2)[1],
                ft1=(ft + [np.nan] * 2)[0], ft2=(ft + [np.nan] * 2)[1],
                **{k: float(cost[k][0]) for k in C.COMPONENTS + ("T", "gtt", "hwt", "gtc")},
                hwt_first=float(C.hidden_wait([freq[ch[0]]], par)[0]), freq=capped,
                freq_counted=counted, basis=basis, n_days=nd)
    return out


def r3_kept(ref, mu):
    """R3 by hand on float32 values (as stored and as rules.apply_rules sees them)."""
    def cls(k):
        return k[0], "".join(sorted(set(ref[k]["seq"])))
    g, h = defaultdict(lambda: np.inf), defaultdict(lambda: np.inf)
    for k, v in ref.items():
        g[cls(k)] = min(g[cls(k)], float(np.float32(v["gtt"])))
        h[cls(k)] = min(h[cls(k)], float(np.float32(v["gtc"])))
    return {k for k, v in ref.items()
            if float(np.float32(v["gtt"])) <= (1 + mu) * g[cls(k)]
            or float(np.float32(v["gtc"])) <= (1 + mu) * h[cls(k)]}


def as_keys(df):
    s = df[df["seq"] != "C"]
    return {(int(d), int(a), int(b), int(c)): r for d, a, b, c, r in
            zip(s["d"], s["l1"], s["l2"], s["l3"], s.to_dict("records"))}


def assert_same(got, want):
    assert set(got) == set(want), (sorted(set(got) - set(want))[:3], sorted(set(want) - set(got))[:3])
    for k, w in want.items():
        g = got[k]
        for col, v in w.items():
            if isinstance(v, str):
                assert g[col] == v, (k, col, g[col], v)
            elif np.isnan(v):
                assert np.isnan(g[col]), (k, col, g[col])
            else:
                assert abs(float(g[col]) - float(v)) <= 1e-4 * max(1.0, abs(float(v))), (k, col, g[col], v)


def ref_alternatives(cands, car_minutes, p):
    """apply_rules on reference candidates of one pair: representative keys,
    status, variants (keys) in order."""
    rows = [dict(key=v["key"], seq=v["seq"], n=v["n"], T=float(np.float32(v["T"])),
                 gtt=float(np.float32(v["gtt"])), gtc=float(np.float32(v["gtc"])),
                 freq=float(np.float32(v["freq"])), first_link=k[1], cities=v["cities"])
            for k, v in cands.items()]
    if car_minutes is not None:
        g = float(np.float32(p["BETA_IVT"]["C"] * car_minutes))
        rows.append(dict(key="car", seq="C", n=0, T=float(np.float32(car_minutes)), gtt=g, gtc=g,
                         freq=np.inf, first_link=-1, cities=""))
    df = pd.DataFrame(rows)
    df["cand"] = np.arange(len(df))
    alts, _ = RU.apply_rules(df, p, guarantee=bool(p["R4_GUARANTEE"]))
    key = df["key"].to_numpy()
    return [(key[a], s, [key[m] for m in mem if m != a][:p["VARIANTS_KEPT"]])
            for a, s, mem in zip(alts["alt"], alts["status"], alts["members"])]


# --- 1. Random networks against the reference ----------------------------------
CASES = [
    (101, {}),
    (102, dict(MU=0.0, MODE_WORDS="oct21")),
    (103, dict(MODE_WORDS="deck35",
               WINDOWS={"AA": [30, 240], "AP": [40, 360], "PA": [60, 360], "PP": [3, 120]})),
    (104, dict(MU=math.inf, T_MAX_MIN=600, MAX_TRANSFERS=1)),
    (105, dict(MIN_ACTIVE_DATES=3, CROSS_CLASS=True, K_CAP=3, K_EXTRA=0, VARIANTS_KEPT=2)),
]


@pytest.mark.parametrize("seed,change", CASES)
def test_random_networks_against_reference(seed, change):
    p = prm(**change)
    tb = contract_tables(*generate(seed, p), p)
    net = network(tb, p)
    car = {(int(o), int(d)): float(m) for o, d, m in zip(tb["car"]["o"], tb["car"]["d"], tb["car"]["minutes"])}
    n3 = 0
    for o in [int(x) for x in net.fuas]:
        ref = reference(tb, o, p)
        n3 += sum(k[3] >= 0 for k in ref)
        assert_same(as_keys(SR.candidates(net, o, None, p)), ref)
        for mu in (0.0, 0.5):
            assert set(as_keys(SR.candidates(net, o, None, p, mu=mu))) == r3_kept(ref, mu)
        mu = p["MU"]
        kept = r3_kept(ref, mu) if math.isfinite(mu) else set(ref)
        out = SR.run_origin(net, o, p)
        best = {}
        for k, v in ref.items():
            cand = (np.float32(v["gtt"]), k[1:])
            best[(k[0], v["seq"])] = min(best.get((k[0], v["seq"]), cand), cand)
        b = out["best"][out["best"]["seq"] != "C"]
        assert {(int(r.d), r.seq): (int(r.l1), int(r.l2), int(r.l3)) for r in b.itertuples()} == \
            {k: v[1] for k, v in best.items()}
        f = out["funnel"].set_index("d")
        a, var = out["alternatives"], out["variants"]
        for d in f.index:
            mine = {k: v for k, v in ref.items() if k[0] == d and k in kept}
            assert f.loc[d, "n_r3"] == len(mine) + ((o, d) in car)
            assert f.loc[d, "n_r0"] >= sum(k[0] == d for k in ref) + ((o, d) in car)
            if not mine and (o, d) not in car:
                assert not (a["d"] == d).any() and not f.loc[d, "pt"]
                continue
            want = ref_alternatives(mine, car.get((o, d)), p)
            got_a = a[a["d"] == d]
            assert got_a["key"].tolist() == [w[0] for w in want]
            assert got_a["status"].tolist() == [w[1] for w in want]
            for rank, (_, _, members) in enumerate(want, 1):
                v = var[(var["d"] == d) & (var["rank"] == rank)].sort_values("member")
                assert v["key"].tolist() == members
            assert f.loc[d, "pt"] == any(w[0] != "car" for w in want)
    assert (n3 > 0) == (p["MAX_TRANSFERS"] == 2)


# --- 2. Cases built by hand ----------------------------------------------------
def hourly(minute, ride):
    return [(60 * h + minute, ride, DAYS) for h in range(6, 22)]


def build(hubs, spec, conn, moves, car, p=P):
    """contract_tables from link specs (src, trg, mode, [(dep, ride, days)])."""
    trip = itertools.count(1)
    links = [[s, t, m, [(e[0], e[1], next(trip), e[2]) for e in evs]] for s, t, m, evs in spec]
    return contract_tables(hubs, links, conn, moves, car, p)


def chains(c):
    """{link ids: row} of the service candidates."""
    return {(int(r.d),) + tuple(int(x) for x in (r.l1, r.l2, r.l3) if x >= 0): r
            for r in c.itertuples() if r.seq != "C"}


def test_moves_airports_and_city_rules_by_hand():
    """Stations o1 (FUA 1), x1 (3), y1, y2 (4; a 5-minute move), d1 (5);
    airports aX (city 3, also reached from 1; a 0-minute colocated move to
    x1), aY (city 4; a 7-minute road move to y2), aZ (no connector: its own
    city) and aD (city 5, also serving 3)."""
    hubs = [("S", 1, "", 1), ("S", 3, "", 1), ("S", 4, "", 1), ("S", 4, "", 1), ("S", 5, "", 1),
            ("A", -1, "XXX", 1), ("A", -1, "YYY", 1), ("A", -1, "ZZZ", 1), ("A", -1, "DDD", 1)]
    o1, x1, y1, y2, d1, aX, aY, aZ, aD = range(9)
    conn = {(1, o1): (10.0, 10.0), (3, x1): (10.0, 10.0), (4, y1): (10.0, 10.0),
            (4, y2): (10.0, 10.0), (5, d1): (10.0, 10.0), (3, aX): (30.0, 30.0),
            (1, aX): (60.0, 60.0), (4, aY): (30.0, 30.0), (5, aD): (30.0, 30.0),
            (3, aD): (100.0, 100.0)}
    moves = {(x1, aX): 0.0, (aX, x1): 0.0, (aY, y2): 7.0, (y2, aY): 7.0,
             (y1, y2): 5.0, (y2, y1): 5.0}
    spec = [(o1, x1, "R", hourly(0, 50)),                       # 0
            (aX, aY, "A", [(600, 60, DAYS)]),                    # 1
            (y2, d1, "R", hourly(25, 40)),                       # 2
            (x1, y1, "R", [(545, 50, DAYS[:20]), (780, 50, DAYS)]),  # 3
            (y1, d1, "R", [(870, 40, DAYS)]),                    # 4
            (aX, aZ, "A", [(700, 60, DAYS)]),                    # 5
            (aZ, aD, "A", [(840, 60, DAYS)])]                    # 6
    tb = build(hubs, spec, conn, moves, [(1, 5, 300.0)])
    got = chains(SR.candidates(network(tb, P), 1, None, P))
    # a move at both transfers, the first one 0 minutes between two hubs
    r = got[(5, 0, 1, 2)]
    assert (r.seq, r.mv1, r.mv2, r.h_first, r.h_last) == ("RAR", 0.0, 7.0, o1, d1)
    assert r.cities == "1>3>XXX>YYY>4>5"
    # a station move at the second transfer; the 09:05 run of link 3 has no
    # onward connection, the 13:00 run has
    assert got[(5, 0, 3, 2)].mv2 == 5.0 and (5, 0, 3, 4) in got
    # an airport without a city as a transfer hub
    assert got[(5, 0, 5, 6)].cities == "1>3>XXX>ZZZ>DDD"
    assert got[(5, 5, 6)].cities == "XXX>ZZZ>DDD"
    # from 1 through aX (city 3): destination 3 is c_first, cut even though
    # aD serves 3
    assert (3, 5, 6) not in got
    assert {(k[0],) + tuple(k[1:]) + (-1,) * (4 - len(k)) for k in got} == set(reference(tb, 1, P))


def test_triple_without_opportunity_although_both_pairs_exist():
    """o1 -> x1 hourly (arrivals at :50); x1 -> y1 at 09:05 (fed) and 03:00
    (no feeder); y1 -> d1 at 04:30 (fed by the 03:00 run only). Both pairs
    have rows; the triple has no opportunity, so it is no candidate."""
    hubs = [("S", 1, "", 1), ("S", 3, "", 1), ("S", 4, "", 1), ("S", 5, "", 1)]
    spec = [(0, 1, "R", hourly(0, 50)), (1, 2, "R", [(545, 50, DAYS), (180, 50, DAYS)]),
            (2, 3, "R", [(270, 40, DAYS)])]
    conn = {(1, 0): (10.0, 10.0), (3, 1): (10.0, 10.0), (4, 2): (10.0, 10.0), (5, 3): (10.0, 10.0)}
    tb = build(hubs, spec, conn, {}, [])
    rows = tb["transfers"].set_index(["link_in", "link_out"])
    assert (0, 1) in rows.index and (1, 2) in rows.index
    net = network(tb, P)
    got = chains(SR.candidates(net, 1, None, P))
    assert (5, 0, 1, 2) not in got and (4, 0, 1) in got and (5, 1, 2) not in got
    out = SR.run_origin(net, 1, P)
    assert not (out["alternatives"]["key"] == "0-1-2").any()


def test_origin_without_access_hub_and_origin_without_paths():
    hubs = [("S", 1, "", 1), ("S", 5, "", 1), ("S", 4, "", 1)]
    spec = [(0, 1, "R", hourly(0, 50))]
    conn = {(1, 0): (10.0, 10.0), (5, 1): (10.0, 10.0), (4, 2): (10.0, 10.0)}
    tb = build(hubs, spec, conn, {}, [(90, 5, 200.0), (90, 1, 100.0), (1, 5, 300.0)])
    net = network(tb, P)
    out = SR.run_origin(net, 90, P)            # FUA 90: car rows only, no access hub
    f = out["funnel"].set_index("d")
    assert sorted(f.index) == [1, 4, 5] and not f["pt"].any()
    assert out["alternatives"]["key"].tolist() == ["car", "car"]
    assert dict(zip(SR.unserved(out["funnel"])["d"], SR.unserved(out["funnel"])["reason"])) == {
        1: "no_service_path", 4: "nothing", 5: "no_service_path"}
    out = SR.run_origin(net, 4, P)             # FUA 4: an access hub without links
    assert len(out["alternatives"]) == 0 and not out["funnel"]["pt"].any()
    out = SR.run_origin(net, 1, P)
    a = out["alternatives"]
    assert a["key"].tolist() == ["0", "car"] or a["key"].tolist() == ["car", "0"]
