"""Static service network of the path-set module (plan step 1).

Builds, from every date of the timeline, the hubs, the service links with
their frequencies, the events (distinct runs) with their calendars, and the
direct car alternative. Rail and coach come from data-upstream-hubs, flights,
car, FUAs and airports from data-upstream. Steps a to i follow the plan's
Definitions; the rules themselves are in pathset/network.py.

Writes to PATHSET_DIR: ps-hubs, ps-links, ps-sparse, ps-events,
ps-event-masks.npy, ps-link-members, ps-drops, ps-car and one record in
ps-manifest.jsonl.

Run:  python -X utf8 pathset-network.py
"""

#%% Imports
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathset import dates as D
from pathset import io
from pathset import network as N
from pathset import schemas as S

P = io.P
INPUTS: list[Path] = []      # every file read, for the manifest
CAR_FASTEST_SHOWN = 10       # rows of the car speed list in the log


def read(name: str, root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    INPUTS.append(Path(root) / f"{name}.parquet")
    return io.load(name, root, columns)


#%% a. Hubs
def step_hubs() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Station hubs and airports; the country offsets set beside
    air-times.py:COUNTRY_TZ."""
    st = read("ic-stations", io.HUBS, ["stn", "name", "fua", "geometry"])
    st["lon"], st["lat"] = N.wkb_lonlat(st.pop("geometry"))
    calls = read("ic-timetable", io.HUBS, ["stn"])["stn"].value_counts()
    st["calls"] = st["stn"].map(calls).fillna(0).to_numpy()
    ap = read("airports", io.DATA, ["iata", "name", "icc", "geometry"])
    ap["lon"], ap["lat"] = N.wkb_lonlat(ap.pop("geometry"))
    fuas = read("fuas", io.DATA, ["id", "icc"])
    INPUTS.append(io.RELEASE / "nodes.csv")
    nodes = pd.read_csv(io.RELEASE / "nodes.csv", encoding="utf-8")
    hubs, stn2hub = N.build_hubs(st, ap, fuas, nodes, P["HUB_RADIUS_M"],
                                 P["XWALK_M"])

    upstream = N.upstream_country_tz(io.ROOT / "air-times.py")
    tz_diff = {c: [N.TZ_STD.get(c), upstream.get(c)]
               for c in sorted(set(N.TZ_STD) | set(upstream))
               if N.TZ_STD.get(c) != upstream.get(c)}
    is_s = hubs["kind"] == "S"
    io.log(f"a. hubs: {int(is_s.sum()):,} station hubs from {len(st):,} "
           f"stations ({int((hubs['stns'].str.len() > 1).sum())} with more "
           f"than one station), {int((~is_s).sum())} airports; released id "
           f"for {int((hubs.loc[is_s, 'released_id'] != '').sum())} station "
           f"hubs and {int((hubs.loc[~is_s, 'released_id'] != '').sum())} "
           "airports")
    io.log("a. country offsets against air-times.py COUNTRY_TZ: "
           + (f"DIFFERENT {tz_diff}" if tz_diff else "identical"))
    return hubs, stn2hub, tz_diff


#%% b. Trips
def route_types(jn: pd.DataFrame) -> np.ndarray:
    """GTFS route type of each journey: ic-journeys (feed, trip) -> db-trips
    (fid, id) -> db-routes (fid, id).mode_id. Only three columns of db-trips
    are read, for the feeds present."""
    INPUTS.append(io.DATA / "gtfs/db-trips.parquet")
    trips = pq.read_table(
        io.DATA / "gtfs/db-trips.parquet", columns=["fid", "id", "route_id"],
        filters=[("fid", "in", np.unique(jn["feed"]).tolist())]).to_pandas()
    routes = (read("gtfs/db-routes", io.DATA, ["fid", "id", "mode_id"])
              .rename(columns={"id": "route_id"}))
    m = (jn[["feed", "trip"]]
         .merge(trips, left_on=["feed", "trip"], right_on=["fid", "id"],
                how="left")
         .merge(routes, on=["fid", "route_id"], how="left"))
    if len(m) != len(jn) or m["mode_id"].isna().any():
        raise ValueError("journeys without a route type in db-trips/db-routes")
    return m["mode_id"].to_numpy(dtype=np.int16)


def step_trips(hubs, stn2hub, drops) -> tuple[N.Trips, pd.DataFrame]:
    """Journeys of the kept feeds -> trips. Returns the trips and, per trip,
    the lines and route types of its journeys."""
    jn = read("ic-journeys", io.HUBS, ["jrn", "line", "dateset", "feed", "trip"])
    feeds = read("feed-info", io.HUBS, ["id", "name"])
    excluded = feeds[feeds["name"].isin(P["FEED_EXCLUDE"])]
    if len(excluded) != len(P["FEED_EXCLUDE"]):
        raise ValueError(f"FEED_EXCLUDE {P['FEED_EXCLUDE']} not all in feed-info")
    out = jn["feed"].isin(excluded["id"])
    drops.append(("b. trips", "feed in FEED_EXCLUDE", int(out.sum()),
                  "journeys; " + ", ".join(
                      f"{n} = feed {i}" for i, n in
                      zip(excluded["id"], excluded["name"]))))
    jn = jn[~out].reset_index(drop=True)
    n_jrn = len(jn)

    lines = read("ic-lines", io.HUBS, ["line", "rail"])
    jn = jn.merge(lines, on="line", how="left", validate="m:1")
    if jn["rail"].isna().any():
        raise ValueError("journeys of lines missing from ic-lines")
    jn["rail"] = jn["rail"].astype(bool)
    jn["route_type"] = route_types(jn)

    ids, masks = N.dateset_masks(read("ic-datesets", io.HUBS))
    jn["cal"] = np.searchsorted(ids, jn["dateset"].to_numpy())
    if (ids[np.minimum(jn["cal"], len(ids) - 1)] != jn["dateset"]).any():
        raise ValueError("journeys with a dateset missing from ic-datesets")

    tt = read("ic-timetable", io.HUBS, ["jrn", "stn", "arr", "dep"])
    tt = tt[tt["jrn"].isin(jn["jrn"])]
    calls = pd.DataFrame({
        "jrn": tt["jrn"].to_numpy(),
        "hub": stn2hub.reindex(tt["stn"]).to_numpy(),
        "arr": tt["arr"].to_numpy(), "dep": tt["dep"].to_numpy()})
    trips = N.build_trips(calls, jn[["jrn", "rail", "route_type", "cal"]],
                          hubs["tz_std"].to_numpy(), masks)

    t = trips.trips
    drops.append(("b. trips", "station call absorbed into one hub call",
                  trips.n_calls_merged, "consecutive calls at one hub"))
    drops.append(("b. trips", "journey equal to an earlier one (merged)",
                  n_jrn - len(t),
                  "same hub sequence and local times; calendars OR-ed"))
    mixed = t["mixed"].to_numpy()
    drops.append(("b. trips", "rail and bus flags in one trip (decided by "
                  "route type)", int(mixed.sum()),
                  f"trips, {int(t.loc[mixed, 'n_jrn'].sum()):,} journeys; "
                  f"{int((t.loc[mixed, 'mode'] == N.MODE_R).sum())} to R"))
    twice = trips.calls.duplicated(["trip", "hub"]).groupby(
        trips.calls["trip"]).any().sum()
    io.log(f"b. trips: {len(t):,} from {n_jrn:,} journeys "
           f"(R {int((t['mode'] == N.MODE_R).sum()):,}, "
           f"B {int((t['mode'] == N.MODE_B).sum()):,}); "
           f"{trips.n_calls_merged:,} station calls absorbed; "
           f"{int(mixed.sum()):,} trips with both flags; "
           f"{int(twice)} trips call at a hub twice")
    jn["trip"] = trips.jrn_trip.reindex(jn["jrn"]).to_numpy()
    trip_lines = jn[["trip", "line", "route_type"]].drop_duplicates()
    return trips, trip_lines


#%% c, d. Legs and hygiene
def screen(legs: pd.DataFrame, hubs: pd.DataFrame, step: str, drops
           ) -> pd.DataFrame:
    """Apply the hygiene rules and log every reason by mode."""
    lon, lat = hubs["lon"].to_numpy(), hubs["lat"].to_numpy()
    src, trg = legs["src"].to_numpy(), legs["trg"].to_numpy()
    code = N.hygiene(legs["ride"], N.gc_km(lon[src], lat[src], lon[trg], lat[trg]),
                     legs["mode"], P["MAX_RIDE_MIN"], P["V_MAX_KMH"])
    by = pd.crosstab(code, N.MODE_LETTERS[legs["mode"].to_numpy()])
    for reason in range(1, len(N.HYGIENE_REASONS)):
        n = by.loc[reason] if reason in by.index else pd.Series(dtype=int)
        drops.append((step, N.HYGIENE_REASONS[reason], int(n.sum()),
                      "legs; " + ", ".join(f"{m} {v:,}" for m, v in n.items()
                                           if v)))
    return legs[code == 0].reset_index(drop=True)


def step_legs(trips: N.Trips, hubs, drops) -> tuple[pd.DataFrame, np.ndarray]:
    """Legs between FUAs, screened. Also returns the hub pairs with a leg
    before the screen."""
    legs, n_same = N.build_legs(trips.calls, trips.trips["mode"].to_numpy(),
                                hubs["fua"].to_numpy(),
                                hubs["tz_std"].to_numpy())
    drops.append(("c. legs", "pair inside one FUA (not built)", n_same,
                  "ordered pairs of calls of a trip"))
    drops.append(("c. legs", "trip without a leg between two FUAs",
                  len(trips.trips) - legs["trip"].nunique(), "trips"))
    pairs = legs[["src", "trg"]].drop_duplicates().to_numpy()
    ok = screen(legs, hubs, "d. hygiene", drops)
    io.log(f"c. legs: {len(legs):,} between FUAs ({n_same:,} pairs inside one "
           f"FUA not built); d. hygiene keeps {len(ok):,}")
    return ok, pairs


#%% e. Runs
def to_runs(legs: pd.DataFrame, calendars, step: str, drops
            ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Equal legs -> runs -> near-duplicates merged. ``calendars(run_of_leg,
    n_runs)`` gives the calendar of each run. Returns (runs, calendars, row of
    the run for every leg; -1 where the run has no date)."""
    runs, run_of_leg = N.merge_equal(legs)
    masks = calendars(run_of_leg, len(runs))
    dated = D.popcount_rows(masks) > 0
    drops.append((step, "leg equal to an earlier one (merged)",
                  len(legs) - len(runs), "same src, trg, mode, dep, ride"))
    drops.append((step, "run without a date in the timeline",
                  int((~dated).sum()), "runs"))
    merged, masks, kept_of = N.merge_near(
        runs[dated].reset_index(drop=True), masks[dated], P["NEAR_DUP_MIN"])
    drops.append((step, "near-duplicate run (merged)",
                  int(dated.sum()) - len(merged),
                  f"dep and ride within {P['NEAR_DUP_MIN']} min, common date"))
    run_of_dated = np.full(len(runs), -1, dtype=np.int64)
    run_of_dated[dated] = kept_of
    io.log(f"{step}: {len(legs):,} legs -> {len(runs):,} runs -> "
           f"{len(merged):,} after near-duplicates")
    return merged, masks, run_of_dated[run_of_leg]


#%% f. Flights
def step_flights(hubs, trip0: int, drops) -> tuple[pd.DataFrame, np.ndarray]:
    """Non-stop flights between study airports as legs with a date range.
    Trip keys start at ``trip0``. Also returns the airport pairs flown."""
    cols = ["carrier", "flight", "src", "trg", "dep", "nstops", "op_days",
            "start_date", "end_date"]
    tt = read("air-timetable", io.DATA, cols)
    hub_of = hubs.loc[hubs["kind"] == "A"].set_index("iata")["hub"]
    keep = ((tt["nstops"] == 0) & tt["src"].isin(hub_of.index)
            & tt["trg"].isin(hub_of.index))
    drops.append(("f. flights", "not non-stop or airport outside the study "
                  "set", int((~keep).sum()), "air-timetable rows"))
    tt = tt[keep].reset_index(drop=True)
    sched = read("air-schedule", io.DATA,
                 ["src", "trg", "carrier", "flight", "start_date", "duration"])
    N.check_air_alignment(tt, sched)

    day0, day1 = N.day_range(tt["start_date"], tt["end_date"])
    fl = pd.DataFrame({
        "src": tt["src"].astype(str).map(hub_of).to_numpy(dtype=np.int32),
        "trg": tt["trg"].astype(str).map(hub_of).to_numpy(dtype=np.int32),
        "mode": N.MODE_A,
        "dep": tt["dep"].to_numpy(dtype=np.int16),
        "ride": sched["duration"].to_numpy(dtype=np.int32),
        "carrier": tt["carrier"].astype(str).to_numpy(),
        "flight": tt["flight"].to_numpy(),
        "day0": day0, "day1": day1,
        "bits": N.op_days_bits(tt["op_days"])})
    fl["trip"] = trip0 + fl.groupby(
        ["carrier", "flight", "src", "trg", "dep"], sort=True).ngroup()
    inside = (fl["day0"] <= fl["day1"]).to_numpy()
    drops.append(("f. flights", "date range outside the timeline",
                  int((~inside).sum()), "records"))
    pairs = fl[["src", "trg"]].drop_duplicates().to_numpy()
    ok = screen(fl[inside].reset_index(drop=True), hubs, "f. flights", drops)
    io.log(f"f. flights: {len(tt):,} non-stop records between study airports, "
           f"{int(inside.sum()):,} inside the timeline, {len(ok):,} after "
           f"hygiene; {fl['trip'].nunique():,} trip keys")
    return ok, pairs


#%% g, h. Classes, links, events, members
def step_links(runs, masks, hubs) -> N.Network:
    cls = N.ride_classes(runs, P["RIDE_CLASS_ABS_MIN"], P["RIDE_CLASS_REL"],
                         P["AIR_ONE_CLASS"])
    fixed = (D.day_index(P["FIXED_FREQ_START"]), D.day_index(P["FIXED_FREQ_END"]))
    net = N.build_links(runs, masks, cls, hubs, P["MIN_ACTIVE_DATES"], fixed)
    for mode in S.MODES:
        k, s = net.links["mode"] == mode, net.sparse["mode"] == mode
        io.log(f"g. links {mode}: {int(k.sum()):,} kept "
               f"({int(net.links.loc[k, 'n_events'].sum()):,} events, median "
               f"freq {net.links.loc[k, 'freq'].median():.1f}), "
               f"{int(s.sum()):,} sparse")
    return net


def step_keys(net, ground, flights) -> None:
    """Final trip keys of ps-events (``N.vehicle_keys``), written into
    ``net.events``. ``ground`` and ``flights`` as in ``step_members``."""
    pairs = [(recs["trip"].to_numpy(),
              np.where(run >= 0, net.event_of_run[np.maximum(run, 0)], -1))
             for recs, run in (ground, flights)]
    trip = np.concatenate([t for t, _ in pairs])
    event = np.concatenate([e for _, e in pairs])
    key = N.vehicle_keys(trip, event, len(net.events))
    before = net.events["trip"].to_numpy()
    net.events["trip"] = key.astype(np.int32)
    on = event >= 0
    size = np.bincount(np.unique(np.c_[trip[on], key[event[on]]], axis=0)[:, 1])
    io.log(f"h. trip keys: {len(np.unique(key)):,} vehicle groups from "
           f"{len(np.unique(trip[on])):,} trips with a written event (largest "
           f"{int(size.max())} trips); {int((before != key).sum()):,} events "
           "re-keyed")


def step_members(net, ground, flights, trip_lines) -> pd.DataFrame:
    """ps-link-members. ``ground`` = (legs, run of each leg), ``flights`` =
    (records, run of each record); run rows are those of the table given to
    ``build_links``."""
    legs, run = ground
    g = pd.DataFrame({"event": net.event_of_run[run[run >= 0]],
                      "trip": legs["trip"].to_numpy()[run >= 0]})
    g = g[g["event"] >= 0].drop_duplicates().merge(trip_lines, on="trip")
    agency = read("ic-lines", io.HUBS, ["line", "agency"]).set_index("line")["agency"]
    g["agency"] = g["line"].map(agency).to_numpy()
    recs, run = flights
    a = pd.DataFrame({"event": net.event_of_run[run[run >= 0]],
                      "agency": recs["carrier"].to_numpy()[run >= 0],
                      "route_type": -1, "line": -1})
    a = a[a["event"] >= 0].drop_duplicates()
    m = pd.concat([g[["event", "agency", "route_type", "line"]], a])
    return N.link_members(m["event"], net.events["link"].to_numpy(),
                          m["agency"], m["route_type"], m["line"])


#%% i. Car
def step_car() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ps-car and its fastest pairs on the straight line between FUA centres."""
    car = read("car-links", io.DATA)
    fuas = read("fuas", io.DATA, ["id", "name", "centre"])
    out = N.build_car(car, fuas.set_index("id")["name"])
    lon, lat = N.wkb_lonlat(fuas["centre"])
    name = fuas.set_index("id")["name"]
    top = out.assign(kmh=N.car_speed(out, fuas["id"], lon, lat)).nlargest(
        CAR_FASTEST_SHOWN, "kmh")
    io.log(f"i. car: {len(out):,} pairs, {int(out['fixed'].sum()):,} with the "
           "crossing restored; fastest on the straight line:")
    for r in top.itertuples():
        io.log(f"   {name[r.o]} -> {name[r.d]}: {r.kmh:.0f} km/h "
               f"({r.minutes:.0f} min)")
    return out, top


#%% Run
def main() -> None:
    t0 = time.time()
    clock = {}
    drops: list[tuple] = []

    hubs, stn2hub, tz_diff = step_hubs()
    clock["a"] = time.time() - t0
    trips, trip_lines = step_trips(hubs, stn2hub, drops)
    clock["b"] = time.time() - t0
    legs, ground_pairs = step_legs(trips, hubs, drops)
    g_runs, g_masks, g_run_of_leg = to_runs(
        legs, lambda run, n: N.or_calendars(
            run, n, legs["trip"], legs["shift"], trips.masks), "e. runs", drops)
    clock["e"] = time.time() - t0
    recs, air_pairs = step_flights(hubs, len(trips.trips), drops)
    a_runs, a_masks, a_run_of_rec = to_runs(
        recs, lambda run, n: N.weekday_calendars(
            run, n, recs["day0"], recs["day1"], recs["bits"]),
        "f. flights", drops)
    clock["f"] = time.time() - t0

    runs = pd.concat([g_runs, a_runs], ignore_index=True)
    masks = np.vstack([g_masks, a_masks])
    a_run_of_rec = np.where(a_run_of_rec >= 0, a_run_of_rec + len(g_runs), -1)
    net = step_links(runs, masks, hubs)
    step_keys(net, (legs, g_run_of_leg), (recs, a_run_of_rec))
    members = step_members(net, (legs, g_run_of_leg), (recs, a_run_of_rec),
                           trip_lines)
    clock["h"] = time.time() - t0

    drops.append(("h. links", "link active on fewer than MIN_ACTIVE_DATES "
                  "dates (to ps-sparse)", len(net.sparse),
                  "links; " + ", ".join(
                      f"{m} {v:,}" for m, v in
                      net.sparse["mode"].value_counts().sort_index().items())))
    drops.append(("h. links", "event of a sparse link (not written)",
                  int((net.event_of_run < 0).sum()), "runs"))
    built = pd.MultiIndex.from_arrays(np.vstack([ground_pairs, air_pairs]).T)
    linked = pd.MultiIndex.from_frame(
        pd.concat([net.links[["src", "trg"]], net.sparse[["src", "trg"]]]))
    lost = built.difference(linked)
    drops.append(("h. links", "hub pair with legs but no link", len(lost),
                  " ".join(f"{s}>{t}" for s, t in lost)))
    io.log(f"h. hub pairs with legs but no link: {len(lost)}")

    car, car_top = step_car()
    clock["i"] = time.time() - t0

    io.save(hubs, "ps-hubs")
    io.save(net.links, "ps-links")
    io.save(net.sparse, "ps-sparse")
    io.save(net.events, "ps-events")
    io.save_npy(net.masks, "ps-event-masks")
    io.save(members, "ps-link-members")
    io.save(N.drops_table(drops), "ps-drops")
    io.save(car, "ps-car")
    io.write_manifest("network", sorted(set(INPUTS)), {
        "seconds": {k: round(v, 1) for k, v in clock.items()},
        "tz_diff": tz_diff,
        "n_ground_trips": len(trips.trips),
        "hubs": hubs["kind"].value_counts().to_dict(),
        "links": net.links["mode"].value_counts().to_dict(),
        "sparse": net.sparse["mode"].value_counts().to_dict(),
        "events": len(net.events),
        "car": len(car),
        "car_fastest_kmh": float(car_top["kmh"].max()),
    })
    io.log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
