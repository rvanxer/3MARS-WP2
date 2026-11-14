"""Stages C and D: report on the full run of pathset-enumerate.py, and the
sensitivity of the rules on two origins (plan, step 6).

    python -X utf8 pathset-stagecd.py
    python -X utf8 pathset-stagecd.py --sensitivity
    python -X utf8 pathset-stagecd.py --log <console log of the stage D run>
    python -X utf8 pathset-stagecd.py --out-dir <folder>

Without --sensitivity the script reads stored tables only (the combined ps-*
path tables, the attrs of the funnel shards, the network tables,
ps-manifest.jsonl, the saved sensitivity tables when present, and for three
open items a few rows of the network inputs) and writes
PATHSET_DIR/stage-cd/stage-cd-report.md, or <folder>/stage-cd-report.md with
--out-dir (to regenerate the report elsewhere and compare; nothing else is
written there). It takes well under a minute.

--sensitivity first runs, for each origin of pathset.stagecd.SENS_ORIGINS
(Wien 367, Katowice 148), one search at MU = 1.0 and every variant of
pathset.stagecd.SENS_VARIANTS on its candidates, checks that the default
variant equals the stored ps-alternatives of the origin and that R3 of the
default keeps ps-funnel n_r3 per pair, and saves stage-cd/sensitivity (one
row per origin and variant), stage-cd/sensitivity-alts (every alternative)
and stage-cd/sensitivity-runs (one row per origin, with the run hash of the
parameters and inputs, stagecd.sens_code_hash of the code, the time it was
written and the memory of the process). One search process at a time (the
network once, then the origins in turn); about ten minutes for the two
origins. A failed check is saved to stage-cd/sensitivity-default-mismatch,
flagged in the report, and the script exits with status 1 after writing the
report.

--log copies the console log of the stage D run to stage-cd/stage-d-console.log;
section 3 takes the wall clock from that copy (without one, from the manifest).

The constants (the deck's Table 1, the well-known pairs, the variants) are in
pathset/stagecd.py and not in pathset-params.yml: a new key there changes the
run hash of pathset-enumerate.py and makes every stored shard look stale.
"""

#%% Imports
import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from pathset import io, search
from pathset import stagecd as SCD


#%% Command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stages C and D: report and sensitivity")
    ap.add_argument("--sensitivity", action="store_true",
                    help="run the searches of the sensitivity origins first (slow)")
    ap.add_argument("--log", default=None,
                    help="console log of the stage D run, copied into stage-cd/")
    ap.add_argument("--out-dir", default=None,
                    help="write the report to this folder instead of stage-cd/ (the "
                         "sensitivity tables and the log copy stay in stage-cd/)")
    return ap.parse_args(argv)


#%% Sensitivity (slow)
def run_sensitivity(origins) -> bool:
    """The sensitivity tables of ``origins``; True when every check passed."""
    params = dict(io.P)
    records = SCD.V.manifest_records()
    run, _ = SCD.run_records(records)
    named = bool(run["info"]["named"]) if run is not None else True
    if run is not None:
        differ = [k for k in params if k not in SCD.ENUMERATE_UNHASHED
                  and str(params.get(k)) != str(run["params"].get(k))]
        if differ:
            io.log(f"WARNING: pathset-params.yml differs from the stage D run in {differ}")
    stamp = {"params_hash": SCD.run_hash(params, named, SCD.input_files()),
             "code_hash": SCD.sens_code_hash()}
    started = time.perf_counter()
    net = search.load_network(params)
    t_net = time.perf_counter() - started
    t = time.perf_counter()
    search.candidates(net, int(origins[0]), [int(origins[0])], params)   # compile
    io.log(f"sensitivity: network {t_net:.1f} s, kernels {time.perf_counter() - t:.1f} s")
    links = io.load_out("ps-links", ["link", "freq", "freq_fixed", "freq_span", "ride"])
    funnel = io.load_out("ps-funnel", ["o", "d", "n_r3"])
    rows, alts, runs, mism = [], [], [], []
    for o in origins:
        stored = io.load(f"o{int(o):03d}", io.OUT / "paths" / "alternatives")
        res = SCD.run_sensitivity(net, int(o), params, links, stored,
                                  funnel[funnel["o"] == int(o)])
        rows += res["rows"]
        alts.append(res["alts"])
        runs.append(res["run"])
        mism.append(res["mismatch"])
        r = res["run"]
        io.log(f"sensitivity {o}: R3 check {r['r3_check']}, default check "
               f"{r['default_check']}, {r['candidates']:,} candidates, search "
               f"{r['search_s']} s")
        del res
    written = time.strftime("%Y-%m-%d %H:%M:%S")
    runs = [{**r, **stamp, "written": written} for r in runs]
    io.save(pd.DataFrame(rows), SCD.SENS_TABLE)
    io.save(pd.concat(alts, ignore_index=True), SCD.SENS_ALTS)
    io.save(pd.DataFrame(runs), SCD.SENS_RUNS)
    bad = pd.concat(mism, ignore_index=True)
    io.save(bad.astype({c: str for c in bad.columns if bad[c].dtype == object}),
            SCD.SENS_MISMATCH)
    ok = all(r["default_check"] == "pass" and r["r3_check"] == "pass" for r in runs)
    if not ok:
        io.log("SENSITIVITY CHECK FAILED: the default variant does not reproduce the "
               f"stored run; see {SCD.SENS_MISMATCH}")
    io.log(f"sensitivity: {time.perf_counter() - started:.0f} s in all")
    return ok


#%% Report
def report(log: str | None, out_dir: str | None = None) -> Path:
    """Write the report (to stage-cd/, or to ``out_dir``); returns its path."""
    copy = io.out_path(SCD.LOG_COPY)
    if log is not None:
        src = Path(log).resolve()
        if src != copy.resolve():
            shutil.copyfile(src, copy)
            io.log(f"copied {src} to {SCD.LOG_COPY}")
    t = time.perf_counter()
    x = SCD.load_inputs(copy if copy.exists() else None)
    ctx = SCD.prepare(x)
    text = SCD.build_report(x, ctx)
    if out_dir is None:
        path = SCD.write_text(text, SCD.REPORT)
    else:
        path = SCD.write_file(text, Path(out_dir) / Path(SCD.REPORT).name)
    io.log(f"report written in {time.perf_counter() - t:.0f} s")
    return path


#%% Run
def run(args: argparse.Namespace) -> bool:
    ok = run_sensitivity(SCD.SENS_ORIGINS) if args.sensitivity else True
    path = report(args.log, args.out_dir)
    print(f"report: {path}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(parse_args()) else 1)
