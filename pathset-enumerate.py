"""Path enumeration: candidate paths and alternatives of every FUA pair.

Plan step 3 ("Generation") and step 4 ("Rules"); the search is
``pathset/search.py``, the tables are the "Paths" part of
``pathset/schemas.py``. Inputs from PATHSET_DIR: ps-hubs, ps-links, ps-events,
ps-event-masks.npy, ps-transfers, ps-connectors, ps-car.

    python -X utf8 pathset-enumerate.py --origins stagec --named
    python -X utf8 pathset-enumerate.py --origins all --resume --combine
    python -X utf8 pathset-enumerate.py --combine

--origins   all | stagec (STAGE_C_ORIGINS) | comma-separated FUA ids
--workers   worker processes (default WORKERS); each builds the network once
--mu        R3 margin (default MU; "inf" switches R3 off)
--passed-share  share of a link's dated departures that must pass a city for
            the link to pass it (R0; search.passed_cities). Default
            PASSED_SHARE of pathset-params.yml; without that key 1 (every
            run, search.passed_share). The value is hashed and recorded in
            the manifest (info "passed_share").
--named     trace the NAMED_PAIRS whose origin is run (ps-trace)
--resume    skip origins whose shards carry the hash of this run
--combine   concatenate the shards into the ps-* path tables and write
            ps-unserved

One shard per origin and table: PATHSET_DIR/paths/<table>/o<ooo>.parquet,
written atomically (pathset.io.save). Every shard carries the run hash in its
metadata (attrs "param_hash"); the funnel shard is written last, so an origin
is complete when all its shards carry the same hash (--resume skips only
those; --combine refuses anything else). A failed origin stops the run with
its error. One ps-manifest record per run.
"""

#%% Imports
import argparse
import hashlib
import json
import math
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from pathset import io, search

# shard folder -> combined table, with its sort order
COMBINED = {
    "alternatives": ("ps-alternatives", ["o", "d", "rank"]),
    "variants": ("ps-variants", ["o", "d", "rank", "member"]),
    "best": ("ps-best-by-sequence", ["o", "d", "seq"]),
    "funnel": ("ps-funnel", ["o", "d"]),
    "seq_counts": ("ps-seq-counts", ["o", "seq"]),
    "trace": ("ps-trace", ["o", "d", "cand"]),
}
# parameters that cannot change a shard
UNHASHED = ("WORKERS", "MEM_CAND_GB", "STAGE_C_ORIGINS", "STAGE_B_EPS_T_ABS",
            "STAGE_B_MU", "STAGE_B_RHO_F", "DECK_TOP_K")


#%% Command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Candidate paths and alternatives per FUA pair")
    ap.add_argument("--origins", help="all | stagec | comma-separated FUA ids")
    ap.add_argument("--workers", type=int, default=int(io.P["WORKERS"]))
    ap.add_argument("--mu", type=float, default=float(io.P["MU"]),
                    help='R3 margin; "inf" switches R3 off')
    ap.add_argument("--passed-share", type=float, default=io.P.get("PASSED_SHARE"),
                    help="PASSED_SHARE (default: pathset-params.yml, else 1)")
    ap.add_argument("--named", action="store_true", help="trace the NAMED_PAIRS")
    ap.add_argument("--resume", action="store_true", help="skip finished origins")
    ap.add_argument("--combine", action="store_true", help="write the ps-* path tables")
    args = ap.parse_args(argv)
    if not args.origins and not args.combine:
        ap.error("give --origins, --combine or both")
    return args


#%% Origins
def select_origins(spec: str, params: dict) -> list:
    """FUA ids to run: every FUA of ps-connectors and ps-car (all), the stage C
    list (stagec) or the ids given; ids outside the network are an error."""
    universe = search.fua_universe(io.load_out("ps-connectors", ["fua"]),
                                   io.load_out("ps-car", ["o", "d"])).tolist()
    if spec == "all":
        return universe
    ids = (params["STAGE_C_ORIGINS"] if spec == "stagec"
           else [int(x) for x in spec.split(",") if x.strip()])
    unknown = sorted(set(ids) - set(universe))
    if unknown:
        raise ValueError(f"FUAs without connectors or car rows: {unknown}")
    return [int(x) for x in ids]


#%% Shards
def shard_name(table: str, o: int) -> str:
    return f"paths/{table}/o{o:03d}"


def input_files() -> list:
    return [io.OUT / f"{n}.parquet" for n in search.INPUTS] + [io.OUT / "ps-event-masks.npy"]


def run_hash(params: dict, named: bool) -> str:
    """Hash of everything a shard depends on: the parameters (but UNHASHED),
    the --named switch and the size and time of the input files."""
    keep = {k: v for k, v in params.items() if k not in UNHASHED}
    blob = json.dumps({"params": keep, "named": named,
                       "inputs": io.fingerprint(input_files())},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_hash(path) -> str | None:
    """The run hash in a shard's metadata (pandas attrs), read from the
    parquet schema only; None without the file or the hash."""
    if not path.exists():
        return None
    meta = pq.read_schema(path).metadata or {}
    return json.loads(meta.get(b"PANDAS_ATTRS", b"{}")).get("param_hash")


def shard_hash(o: int) -> str | None:
    """The run hash of origin o when every table's shard carries the same
    one, else None."""
    hashes = {file_hash(io.out_path(shard_name(t, o) + ".parquet")) for t in COMBINED}
    return hashes.pop() if len(hashes) == 1 else None


def save_shards(o: int, tables: dict, h: str) -> None:
    """The six tables of one origin, each with the run hash; the funnel
    last."""
    for table in [t for t in COMBINED if t != "funnel"] + ["funnel"]:
        tables[table].attrs["param_hash"] = h
        io.save(tables[table], shard_name(table, o))


#%% Combine
def combine() -> dict:
    """ps-alternatives, ps-variants, ps-best-by-sequence, ps-funnel,
    ps-seq-counts, ps-trace from the shards, and ps-unserved from the funnel.
    Refuses an origin whose shards do not all carry one hash (an interrupted
    save) and shards of different runs."""
    origins = {f.stem for t in COMBINED
               for f in (io.OUT / "paths" / t).glob("o*.parquet")}
    hashes = {}
    for o in sorted(int(stem[1:]) for stem in origins):
        h = shard_hash(o)
        if h is None:
            raise RuntimeError(f"origin {o}: shards missing or of different runs "
                               "(interrupted save?): rerun the origin")
        hashes[o] = h
    if len(set(hashes.values())) > 1:
        raise RuntimeError(f"shards of {len(set(hashes.values()))} different runs: "
                           "rerun the stale origins or remove their shards")
    sizes = {}
    for table, (name, order) in COMBINED.items():
        folder = io.OUT / "paths" / table
        frames = [io.load(f.stem, folder) for f in sorted(folder.glob("o*.parquet"))]
        frames = [f for f in frames if len(f)]
        df = (search.as_table(pd.concat(frames, ignore_index=True), table)
              if frames else search.empty_table(table))
        df = df.sort_values(order, kind="stable").reset_index(drop=True)
        io.save(df, name)
        sizes[name] = len(df)
        if table == "funnel":
            unserved = search.unserved(df)
            io.save(unserved, "ps-unserved")
            sizes["ps-unserved"] = len(unserved)
    return sizes


#%% Run
def origin_line(o: int, tables: dict) -> str:
    """One log line: candidates per pass, triple_stats calls, seconds, memory
    of the worker."""
    st = tables["funnel"].attrs["search"]
    f = tables["funnel"]
    parts = [f"o {o}: r0 {int(f['n_r0'].sum()):,} (by bounds "
             f"{int(f['n_r0_bound'].sum()):,}) r3 {int(f['n_r3'].sum()):,} "
             f"kept {int(f['n_kept'].sum()):,} in {int((f['n_kept'] > 0).sum())} "
             f"destinations"]
    for name in ("pass1", "pass2", "trace_pass2"):
        if name in st:
            p = st[name]
            parts.append(f"{name} {p['s']:.1f}s chains {p['chains1']:,}/{p['chains2']:,}/"
                         f"{p['chains3']:,} eval {p['evaluated']:,} "
                         f"triple {p['triple_calls']:,} cache {p['cache_entries']:,}/"
                         f"{p['cache_cleared']} r0_by_bounds {p['r0_by_bounds']:,}")
    parts.append(f"rules {st['rules_s']:.1f}s total {st['total_s']:.1f}s blocks {st['blocks']} "
                 f"buffer {st['buffer_mb']} MB; memory at start {st['rss_start_mb']} MB, "
                 f"peak {st['peak_mb']} MB")
    return "; ".join(parts)


def run(args: argparse.Namespace) -> dict:
    """Run the origins of ``args`` and/or combine; returns what was done."""
    params = dict(io.P)
    params["MU"] = args.mu
    if args.passed_share is not None:
        params["PASSED_SHARE"] = float(args.passed_share)
    share = search.passed_share(params)
    done, skipped = [], []
    totals = {"r0": 0, "r0_bound": 0, "r3": 0, "kept": 0, "triple_calls": 0, "peak_mb": {}}
    started = time.time()
    if args.origins:
        origins = select_origins(args.origins, params)
        h = run_hash(params, args.named)
        todo = [o for o in origins if not (args.resume and shard_hash(o) == h)]
        skipped = sorted(set(origins) - set(todo))
        io.log(f"enumerate: {len(todo)} origins to run, {len(skipped)} finished, "
               f"{args.workers} workers, MU {params['MU']}, PASSED_SHARE {share}, hash {h}")
        for o, tables in search.run_pool(todo, params, args.workers, named=args.named):
            save_shards(o, tables, h)
            io.log(origin_line(o, tables))
            f, st = tables["funnel"], tables["funnel"].attrs["search"]
            totals["r0"] += int(f["n_r0"].sum())
            totals["r0_bound"] += int(f["n_r0_bound"].sum())
            totals["r3"] += int(f["n_r3"].sum())
            totals["peak_mb"][o] = st["peak_mb"]
            totals["kept"] += int(f["n_kept"].sum())
            totals["triple_calls"] += sum(v["triple_calls"] for v in st.values()
                                          if isinstance(v, dict))
            done.append(o)
    sizes = combine() if args.combine else {}
    info = {"origins_run": done, "origins_skipped": skipped, "workers": args.workers,
            "mu": args.mu if math.isfinite(args.mu) else "inf", "passed_share": share,
            "named": args.named,
            "totals": totals, "combined": sizes, "seconds": round(time.time() - started, 1)}
    io.write_manifest("enumerate", [p for p in input_files() if p.exists()], info)
    return info


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
