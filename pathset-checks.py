"""Checks of the path-set outputs (plan, Verification 2). Network part
(1-5); connectors and moves (6-9).

Prints a short report and exits non-zero when a check fails. Expected values
are those of the plan; a miss is reported with the numbers that explain it,
the rule is not bent to meet it.

Run:  python -X utf8 pathset-checks.py
"""

#%% Imports
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from pathset import connect as CN
from pathset import dates as D
from pathset import io
from pathset import network as N

P = io.P

# Plan, Verification 2: pair-level medians (rail, hub pair, all classes).
EXPECTED_MEDIANS = [
    ("København H", "Malmö C", 25, 30),
    ("Frankfurt (Main) Hauptbahnhof", "Köln Hbf", 35, 42),
    ("Paris Gare de Lyon", "Lyon Part Dieu", 18, 22),
]
# Plan, Verification 2: "no car pair above 110 km/h straight-line".
CAR_V_MAX_KMH = io.P["CAR_V_MAX_KMH"]
CAR_FASTEST_SHOWN = 10
# Services whose local departures are printed for comparison with public
# timetables: (from, to, mode); stations by a name of ic-stations, airports
# by IATA code.
SERVICES = [
    ("Bruxelles-Midi", "St-Pancras-International", "R"),
    ("Amsterdam Centraal", "Berlin Hbf", "R"),
    ("Paris Gare de Lyon", "Lyon Part Dieu", "R"),
    ("Frankfurt (Main) Hauptbahnhof", "Köln Hbf", "R"),
    ("London Kings Cross Rail Station", "Edinburgh Rail Station", "R"),
    ("Stockholm Centralstation", "Malmö C", "R"),
    ("MADRID-PUERTA DE ATOCHA", "Barcelona Sants", "R"),
    ("Basel SBB", "Zürich HB", "R"),
    ("Berlin ZOB", "Hamburg ZOB", "B"),
    ("AMS", "LHR", "A"),
]
LIST_WEEKDAY = 1             # Tuesday (Monday = 0)
LIST_MAX_TIMES = 48          # departures printed per service


#%% Network tables
class Net:
    """The written network, loaded once."""

    def __init__(self):
        self.hubs = io.load_out("ps-hubs")
        self.links = io.load_out("ps-links")
        self.sparse = io.load_out("ps-sparse")
        self.events = io.load_out("ps-events")
        self.masks = io.load_npy("ps-event-masks")
        self.drops = io.load_out("ps-drops")
        self.car = io.load_out("ps-car")
        self.stations = io.load("ic-stations", io.HUBS, ["stn", "name", "fua"])
        s = self.hubs[self.hubs["kind"] == "S"].explode("stns")
        self.stn2hub = pd.Series(s["hub"].to_numpy(),
                                 index=s["stns"].to_numpy(dtype=np.int64))

    def hub(self, name: str) -> int:
        """Hub of the airport with this IATA code, else of the station that
        carries exactly this name among its names; must be unique."""
        air = self.hubs.loc[self.hubs["iata"] == name, "hub"]
        if len(air) == 1:
            return int(air.iloc[0])
        has = self.stations["name"].str.split(" | ", regex=False).apply(
            lambda names: name in names)
        found = set(self.stn2hub.reindex(self.stations.loc[has, "stn"]))
        if len(found) != 1:
            raise KeyError(f"{name!r} names {len(found)} hubs")
        return int(found.pop())

    def pair_events(self, src: int, trg: int, mode: str) -> np.ndarray:
        """Rows of ps-events of all classes of one hub pair and mode."""
        l = self.links[(self.links["src"] == src) & (self.links["trg"] == trg)
                       & (self.links["mode"] == mode)]
        return np.concatenate(
            [np.arange(a, b) for a, b in zip(l["ev0"], l["ev1"])]
            + [np.empty(0, dtype=np.int64)]).astype(np.int64)

    def daily(self, rows: np.ndarray) -> np.ndarray:
        """Departures per day of the timeline over the given events."""
        count = np.zeros(D.N_DAYS, dtype=np.int32)
        for r in rows:
            N.add_daily_counts(self.masks[r], count)
        return count


def hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


#%% 1. Pair-level medians
def check_medians(net: Net) -> bool:
    """Median over active dates of the de-duplicated departures that date,
    per hub pair, rail, all classes together (upstream's pt-links.py rule on
    runs instead of journeys)."""
    print("1. Pair-level medians, rail, all classes")
    ok = True
    for a, b, lo, hi in EXPECTED_MEDIANS:
        count = net.daily(net.pair_events(net.hub(a), net.hub(b), "R"))
        med, active, most = N.median_active(count)
        hit = lo <= med <= hi
        ok &= hit
        print(f"   {a} -> {b}: median {med:g} on {active} active dates "
              f"(expected {lo}-{hi}) {'ok' if hit else 'MISS'}")
        if not hit:
            days = np.flatnonzero(count)
            pos = count[days]
            q = np.percentile(pos, [0, 25, 75, 100]) if active else [0] * 4
            print(f"      daily count min/25%/75%/max = {q[0]:g}/{q[1]:g}/"
                  f"{q[2]:g}/{q[3]:g}; dates {D.index_date(days[0])} to "
                  f"{D.index_date(days[-1])}")
            years = pd.Series(pos).groupby(
                [D.index_date(d).year for d in days]).agg(["median", "size"])
            print("      median by year (active dates): " + ", ".join(
                f"{y}: {r['median']:g} ({int(r['size'])})"
                for y, r in years.iterrows()))
    return ok


#%% 2. Coverage of the station pairs of data-upstream-hubs
def station_pairs(net: Net) -> pd.DataFrame:
    """Directed hub pairs behind every inter-FUA station pair of the lines of
    data-upstream-hubs that have a journey in a kept feed. Read from ic-lines,
    not from the timetable expansion of the build."""
    feeds = io.load("feed-info", io.HUBS, ["id", "name"])
    out = feeds.loc[feeds["name"].isin(P["FEED_EXCLUDE"]), "id"]
    jn = io.load("ic-journeys", io.HUBS, ["line", "feed"])
    kept = set(jn.loc[~jn["feed"].isin(out), "line"])
    lines = io.load("ic-lines", io.HUBS, ["line", "stn"])
    fua = net.stations.set_index("stn")["fua"].to_dict()
    pairs = {(a, b) for line, seq in zip(lines["line"], lines["stn"])
             if line in kept
             for a, b in combinations([int(s) for s in seq], 2)
             if fua[a] != fua[b]}
    p = pd.DataFrame(sorted(pairs), columns=["stn_a", "stn_b"])
    p["src"] = net.stn2hub.reindex(p["stn_a"]).to_numpy()
    p["trg"] = net.stn2hub.reindex(p["stn_b"]).to_numpy()
    return p


def check_coverage(net: Net) -> bool:
    """Every directed inter-FUA station pair is in ps-links, in ps-sparse, or
    listed in the ps-drops row "hub pair with legs but no link"."""
    print("2. Station pairs of data-upstream-hubs (excluded feeds removed)")
    p = station_pairs(net)
    key = pd.MultiIndex.from_frame(p[["src", "trg"]])
    in_links = key.isin(pd.MultiIndex.from_frame(net.links[["src", "trg"]]))
    in_sparse = key.isin(pd.MultiIndex.from_frame(net.sparse[["src", "trg"]]))
    lost = net.drops.loc[
        net.drops["reason"] == "hub pair with legs but no link", "detail"]
    listed = {tuple(map(int, tok.split(">")))
              for tok in " ".join(lost).split()}
    in_drops = np.array([k in listed for k in key], dtype=bool)
    missing = ~(in_links | in_sparse | in_drops)
    print(f"   {len(p):,} station pairs = {key.nunique():,} hub pairs: "
          f"{int(in_links.sum()):,} in ps-links, "
          f"{int((in_sparse & ~in_links).sum()):,} only in ps-sparse, "
          f"{int((in_drops & ~in_links & ~in_sparse).sum()):,} in ps-drops, "
          f"{int(missing.sum()):,} unaccounted")
    for r in p[missing].head(5).itertuples():
        print(f"      unaccounted: stn {r.stn_a} -> {r.stn_b} "
              f"(hub {r.src} -> {r.trg})")
    return not missing.any()


#%% 3. Car
def check_car(net: Net) -> bool:
    """No car pair above CAR_V_MAX_KMH on the straight line between FUA
    centres; the ten fastest are listed."""
    print("3. Car")
    fuas = io.load("fuas", io.DATA, ["id", "name", "centre"])
    lon, lat = N.wkb_lonlat(fuas["centre"])
    name = fuas.set_index("id")["name"]
    car = net.car.assign(kmh=N.car_speed(net.car, fuas["id"], lon, lat))
    top = car.nlargest(CAR_FASTEST_SHOWN, "kmh")
    ok = bool(top["kmh"].max() <= CAR_V_MAX_KMH) and bool(
        (car["minutes"] > 0).all())
    print(f"   {len(car):,} pairs, {int(car['fixed'].sum()):,} with a "
          f"crossing restored; fastest {top['kmh'].max():.0f} km/h (limit "
          f"{CAR_V_MAX_KMH}) {'ok' if ok else 'FAIL'}")
    print("   fastest: " + "; ".join(
        f"{name[r.o]} -> {name[r.d]} {r.kmh:.0f}" for r in top.itertuples()))
    for a, b, *_ in N.CAR_CROSSINGS:
        o, d = name[name == a].index[0], name[name == b].index[0]
        r = car[(car["o"] == o) & (car["d"] == d)].iloc[0]
        print(f"   {a} -> {b}: {r.minutes:.0f} min, {r.km:.0f} km, "
              f"fixed = {bool(r.fixed)}")
    return ok


#%% 4. Local departure lists
def check_departures(net: Net) -> bool:
    """Local departures of well-known services on one weekday, for comparison
    with public timetables. The date is the active day of that weekday whose
    number of departures is nearest to the median of such days (latest if
    tied); a service that never runs on that weekday is shown on another of
    its active dates. Informative: only a station or airport that cannot be
    looked up fails."""
    print(f"4. Local departures, weekday {LIST_WEEKDAY} (Monday = 0)")
    weekday = D.weekdays()
    ok = True
    for a, b, mode in SERVICES:
        try:
            rows = net.pair_events(net.hub(a), net.hub(b), mode)
        except KeyError as err:
            print(f"   {a} -> {b}: {err} FAIL")
            ok = False
            continue
        count = net.daily(rows)
        if not count.any():
            print(f"   {a} -> {b} [{mode}]: no link in ps-links")
            continue
        days = np.flatnonzero((count > 0) & (weekday == LIST_WEEKDAY))
        note = ""
        if len(days) == 0:
            days = np.flatnonzero(count)
            note = f" (never on that weekday; {len(days)} active dates)"
        gap = np.abs(count[days] - np.median(count[days]))
        day = int(days[gap == gap.min()].max())
        on = [r for r in rows if D.get_bit(net.masks[r], day)]
        ev = net.events.iloc[on].sort_values(["dep", "ride"])
        times = [hhmm(int(m)) for m in ev["dep"]]
        more = len(times) - LIST_MAX_TIMES
        print(f"   {a} -> {b} [{mode}] {D.index_date(day):%a %Y-%m-%d}{note}: "
              f"{len(times)} departures, ride {int(ev['ride'].min())}-"
              f"{int(ev['ride'].max())} min")
        print("      " + " ".join(times[:LIST_MAX_TIMES])
              + (f" ... (+{more})" if more > 0 else ""))
    return ok


#%% 5. Totals
def check_totals(net: Net) -> bool:
    """Counts of the written tables and their internal consistency."""
    print("5. Totals")
    l, e = net.links, net.events
    print("   hubs by kind: " + ", ".join(
        f"{k} {v:,}" for k, v in net.hubs["kind"].value_counts().items()))
    print("   links by mode: " + ", ".join(
        f"{k} {v:,}" for k, v in l["mode"].value_counts().sort_index().items())
        + f"; events {len(e):,}; sparse links {len(net.sparse):,}")
    print("   drops by reason:")
    for r in net.drops.itertuples():
        print(f"      {r.step} | {r.reason}: {r.n:,}")
    ok = (len(net.masks) == len(e)
          and bool((l["ev1"] - l["ev0"] == l["n_events"]).all())
          and int(l["n_events"].sum()) == len(e)
          and bool((np.repeat(l["link"].to_numpy(), l["n_events"])
                    == e["link"].to_numpy()).all())
          and bool((D.popcount_rows(net.masks) == e["n_dates"]).all())
          and bool((l["active_dates"] >= P["MIN_ACTIVE_DATES"]).all())
          and bool((net.sparse["active_dates"] < P["MIN_ACTIVE_DATES"]).all())
          and bool(e["dep"].between(0, 1439).all())
          and bool(e["ride"].between(1, P["MAX_RIDE_MIN"]).all()))
    print(f"   tables consistent with the contract: {'ok' if ok else 'FAIL'}")
    return ok


#%% 6. Connectors and moves (plan step 2)
BUSIEST_AIRPORTS = 15        # airports listed by number of links in ps-links


class Conn:
    """ps-connectors, ps-moves and their inputs, loaded once."""

    def __init__(self, net: Net):
        self.hubs, self.links = net.hubs, net.links
        self.conn = io.load_out("ps-connectors")
        self.moves = io.load_out("ps-moves")
        self.fuas = io.load("fuas", io.DATA, ["id", "name", "centre"])
        nodes = pd.read_csv(io.RELEASE / "nodes.csv",
                            usecols=["node_id", "lon", "lat"])
        links = pd.read_csv(io.RELEASE / "links.csv",
                            usecols=["src", "trg", "kind", "time"])
        self.rel = CN.released_tables(nodes, links)
        self.kind = self.hubs.set_index("hub")["kind"]
        self.name = self.hubs.set_index("hub")["name"]


def check_connectors(c: Conn) -> bool:
    """FUAs without an access hub (listed, not failed); released against
    fitted access of station hubs; the table against the contract: one row
    per (fua, hub), station minutes finite, every minute positive or NaN,
    every station hub exactly once and to its own FUA, airport rows with at
    least one finite direction and each finite direction at most
    ACCESS_MAX_AIR_MIN (connect.airport_connectors: NaN above the cap)."""
    print("6. Connectors")
    t = c.conn
    fua_name = c.fuas.set_index("id")["name"]
    none = sorted(set(c.fuas["id"]) - set(t["fua"]))
    print(f"   FUAs without an access hub: {len(none)}"
          + (": " + ", ".join(fua_name[f] for f in none) if none else ""))
    kind = t["hub"].map(c.kind)
    for k, label in (("S", "station hubs"), ("A", "airport connectors")):
        src = t.loc[kind == k, "source"].value_counts()
        n = int(src.sum())
        print(f"   {label}: {n:,}; released {src.get('released', 0):,} "
              f"({src.get('released', 0) / max(n, 1):.1%}), fitted "
              f"{src.get('fitted', 0):,}")
    s = c.hubs[c.hubs["kind"] == "S"]
    st = t[kind == "S"]
    q = st.groupby("source")["acc_min"].quantile([0.5, 1.0]).unstack()
    print("   station access minutes median/max: " + ", ".join(
        f"{k} {r[0.5]:.1f}/{r[1.0]:.1f}" for k, r in q.iterrows()))
    a = t[kind == "A"]
    per_air = a.groupby("hub").size()
    print(f"   airports with a connector: {len(per_air)} of "
          f"{int((c.hubs['kind'] == 'A').sum())}; FUAs per airport median "
          f"{per_air.median():g}, max {per_air.max()}; access only "
          f"{int(a['egr_min'].isna().sum())}, egress only "
          f"{int(a['acc_min'].isna().sum())} (other direction above "
          f"{P['ACCESS_MAX_AIR_MIN']} min)")
    mins = t[["acc_min", "egr_min"]]
    am = a[["acc_min", "egr_min"]]
    ok = (not t.duplicated(["fua", "hub"]).any()
          and bool(np.isfinite(st[["acc_min", "egr_min"]]).all().all())
          and bool(((mins > 0) | mins.isna()).all().all())
          and len(st) == len(s) and set(st["hub"]) == set(s["hub"])
          and bool((st["fua"].to_numpy()
                    == st["hub"].map(s.set_index("hub")["fua"]).to_numpy()).all())
          and bool(am.notna().any(axis=1).all())
          and bool(((am <= P["ACCESS_MAX_AIR_MIN"]) | am.isna()).all().all()))
    print(f"   table consistent with the contract: {'ok' if ok else 'FAIL'}")
    return ok


def check_moves(c: Conn) -> bool:
    """Moves by kind; caps; station moves inside one FUA; co-located moves
    of 0 minutes; no airport-to-airport move. Airport-station moves whose
    station hub lies outside hub_city of the airport are counted, not
    failed (connect.airport_moves: the station may be in any catchment FUA
    of the airport; the contract wording "of one city" awaits a decision)."""
    print("7. Moves")
    m = c.moves
    print("   by kind: " + ", ".join(
        f"{k} {n:,}" for k, n in m["kind"].value_counts().items()))
    ka, kb = m["hub_a"].map(c.kind), m["hub_b"].map(c.kind)
    fua = c.hubs.set_index("hub")["fua"]
    ss = (ka == "S") & (kb == "S")
    air = (ka == "A") | (kb == "A")
    city = CN.hub_city(c.hubs, c.conn)
    a2s = (ka == "A") & (kb == "S")
    cross = a2s & (city[m["hub_a"].to_numpy()] != city[m["hub_b"].to_numpy()])
    n_cross = m.loc[cross, "kind"].value_counts()
    print(f"   airport -> station moves outside the airport's city: "
          f"{int(cross.sum()):,} of {int(a2s.sum()):,} ("
          + ", ".join(f"{k} {n:,}" for k, n in n_cross.items()) + ")")
    ok = (not m.duplicated(["hub_a", "hub_b"]).any()
          and bool((m["hub_a"] != m["hub_b"]).all())
          and not ((ka == "A") & (kb == "A")).any()
          and bool((m.loc[ss, "hub_a"].map(fua).to_numpy()
                    == m.loc[ss, "hub_b"].map(fua).to_numpy()).all())
          and set(m.loc[ss, "kind"]) <= {"interhub", "fitted"}
          and set(m.loc[air, "kind"]) <= {"colocated", "road"}
          and bool((m.loc[ss, "minutes"] <= P["MOVE_MAX_MIN"]).all())
          and bool((m.loc[m["kind"] == "road", "minutes"]
                    <= P["AIR_ROAD_MOVE_MAX_MIN"]).all())
          and bool((m.loc[m["kind"] == "colocated", "minutes"] == 0).all()))
    print(f"   caps, kinds and cities: {'ok' if ok else 'FAIL'}")
    return ok


def check_airports(c: Conn) -> bool:
    """Airports with a co-located station hub, and the busiest airports by
    number of links in ps-links with their co-located hubs and road moves
    (plan, Open: road moves at the busiest airports without a station)."""
    print("8. Airports and stations")
    m = c.moves
    air = c.hubs[c.hubs["kind"] == "A"].set_index("hub")
    col = m[(m["kind"] == "colocated") & m["hub_a"].isin(air.index)]
    road = m[(m["kind"] == "road") & m["hub_a"].isin(air.index)]
    print(f"   airports with a co-located station hub: "
          f"{col['hub_a'].nunique()} of {len(air)}; with a road move only: "
          f"{len(set(road['hub_a']) - set(col['hub_a']))}")
    n_links = (pd.concat([c.links["src"], c.links["trg"]]).value_counts()
               .reindex(air.index, fill_value=0))
    city = CN.hub_city(c.hubs, c.conn)
    fua_name = c.fuas.set_index("id")["name"]
    for h, n in n_links.nlargest(BUSIEST_AIRPORTS).items():
        co = ", ".join(c.name[s] for s in col.loc[col["hub_a"] == h, "hub_b"])
        r = road[road["hub_a"] == h]
        cty = fua_name.get(city[h], "none")
        print(f"   {air.loc[h, 'iata']} ({cty}) {n:,} links: co-located "
              f"[{co or '-'}]; road {len(r)}"
              + (f" ({r['minutes'].min():.0f}-{r['minutes'].max():.0f} min)"
                 if len(r) else ""))
    return True


def check_fits(c: Conn) -> bool:
    """The straight-line fits of the fitted connectors and moves, with their
    leave-one-out error (informative)."""
    print("9. Fits on the released links, minutes = a + b x km")
    fits = CN.released_fits(c.rel, CN.fua_centres(c.fuas))
    for k, f in fits.items():
        print(f"   {k}: a {f.a:.2f}, b {f.b:.3f}, R2 {f.r2:.3f}, MAE "
              f"{f.mae:.1f}, leave-one-out MAE {f.loo_mae:.1f} min, n {f.n}")
    return True


#%% Run
def main() -> int:
    net = Net()
    results = {
        "1 medians": check_medians(net),
        "2 coverage": check_coverage(net),
        "3 car": check_car(net),
        "4 departures": check_departures(net),
        "5 totals": check_totals(net),
    }
    conn = Conn(net)
    results |= {
        "6 connectors": check_connectors(conn),
        "7 moves": check_moves(conn),
        "8 airports": check_airports(conn),
        "9 fits": check_fits(conn),
    }
    failed = [k for k, ok in results.items() if not ok]
    print("FAILED: " + ", ".join(failed) if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
