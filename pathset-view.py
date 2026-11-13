"""Viewer: the funnel, the alternatives and the removed alternatives of one
FUA pair (plan step 5).

    python -X utf8 pathset-view.py "Amsterdam" "Berlin"
    python -X utf8 pathset-view.py 7 28 --order october --r2-time weighted
    python -X utf8 pathset-view.py Amsterdam Berlin --r1-key pattern --eps 20
    python -X utf8 pathset-view.py Amsterdam Muenchen --r2-gate pooled

Origin and destination: a FUA id or name (case and accent insensitive; an
ambiguous name is refused with the candidates listed). The pair is enumerated
again at MU = inf (every R0 candidate), then the rules run with the
parameters of pathset-params.yml, changed by the options. The R0 clause
PASSED_SHARE is --passed-share, else the yml value, else the value of the
latest full run in ps-manifest (view.choose_passed_share), so that the funnel
matches ps-funnel; the header prints the value and its source. Printed: the
funnel (R0 -> R3 -> R1 -> R2 -> cap, or the October order), the kept
alternatives with the deck's components (t_a, t_dep, t_ivt, t_trans, t_tp,
t_arr, t_e: weighted minutes that sum to gtt), T (clock minutes), gtt, hwt,
gtc and freq, then the alternatives removed after R1 per rule with the
numbers of the test, and the R3 cuts per mode sequence.
"""

#%% Imports
import argparse
import sys
import time

import pandas as pd

from pathset import io, search
from pathset import view as V


#%% Command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Alternatives of one FUA pair")
    ap.add_argument("origin", help="FUA id or name")
    ap.add_argument("destination", help="FUA id or name")
    ap.add_argument("--order", choices=["r3_first", "october"], default="r3_first")
    ap.add_argument("--r2-time", choices=["clock", "weighted"], default=io.P["R2_TIME"])
    ap.add_argument("--r1-key", choices=["sequence", "class", "pattern"],
                    default=io.P["R1_KEY"])
    ap.add_argument("--r2-gate", choices=["top", "path", "pooled"],
                    default=io.P["R2_GATE"],
                    help="frequency of the removed alternative in the gate of R2")
    ap.add_argument("--freq", choices=["path", "fixed", "span", "first"], default="path",
                    help="frequency in place of the path frequency (stage B variant 6)")
    ap.add_argument("--mu", type=float, default=float(io.P["MU"]))
    ap.add_argument("--rho", type=float, default=float(io.P["RHO_F"]))
    ap.add_argument("--eps", type=float, default=float(io.P["EPS_T_ABS"]),
                    help="EPS_T_ABS, minutes")
    ap.add_argument("--passed-share", type=float, default=None,
                    help="PASSED_SHARE (default: yml, else the latest full run)")
    ap.add_argument("--max-rows", type=int, default=20,
                    help="removed alternatives printed at most per rule")
    return ap.parse_args(argv)


#%% Run
def run(args: argparse.Namespace) -> dict:
    fuas = V.load_fuas()
    o, d = V.resolve_fua(args.origin, fuas), V.resolve_fua(args.destination, fuas)
    share, source = V.choose_passed_share(args.passed_share, io.P, V.manifest_records())
    params = {**io.P, "R2_TIME": args.r2_time, "R1_KEY": args.r1_key, "MU": args.mu,
              "RHO_F": args.rho, "EPS_T_ABS": args.eps, "PASSED_SHARE": share,
              "R2_GATE": args.r2_gate}
    spec = {"order": args.order, "freq": args.freq}
    started = time.perf_counter()
    net = search.load_network(params)
    links = io.load_out("ps-links", ["link", "freq", "freq_fixed", "freq_span", "ride"])
    names = V.hub_names(io.load_out("ps-hubs", ["hub", "kind", "name", "iata"]))
    cands = V.pair_candidates(net, o, d, params, links)
    t_enum = time.perf_counter() - started
    alts, trace = V.decide_variant(cands, params, spec)
    shown = V.with_frequency(cands, params, args.freq)

    print(f"{V.fua_name(fuas, o)} ({o}) -> {V.fua_name(fuas, d)} ({d}); order "
          f"{args.order}, R1_KEY {args.r1_key}, R2_TIME {args.r2_time}, R2_GATE "
          f"{args.r2_gate}, freq "
          f"{args.freq}, MU {args.mu:g}, RHO_F {args.rho:g}, EPS_T_ABS {args.eps:g}, "
          f"PASSED_SHARE {share:g} ({source}); network and enumeration {t_enum:.1f} s")
    steps = V.funnel(trace, alts, args.order)
    print("funnel: " + " -> ".join(f"{s} {n:,}" for s, n in steps))
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print("\nkept alternatives (t_a .. t_e weighted minutes, sum gtt; T clock "
              "minutes; hwt = gtc - gtt; rides: riding minutes per link)")
        tab = V.alternatives_table(shown, alts, names)
        print(V.fmt_table(tab.drop(columns=["cities"])))
        rem = V.removed_table(shown, trace, names)
        head = V.removed_head(rem, args.max_rows)
        print(f"\nremoved after R1: {len(rem)} ({V.rule_counts(rem)})"
              + (f"; first {args.max_rows} per rule, in increasing gtc"
                 if len(head) < len(rem) else ""))
        if len(head):
            print(V.fmt_table(head[["seq", "route", "T", "gtt", "gtc", "freq", "rule",
                                    "detail", "by_route"]]))
        r3 = V.r3_summary(shown, trace)
        if len(r3):
            print(f"\ncut by R3: {int(r3['n_cut'].sum()):,} candidates")
            print(V.fmt_table(r3))
    return {"o": o, "d": d, "funnel": steps, "kept": len(alts), "passed_share": share}


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
