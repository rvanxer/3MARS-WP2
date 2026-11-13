"""Stage B: the named pairs, every R0 candidate stored, and the rules side by
side with their variants and the deck's top-5 selection (plan step 6).

    python -X utf8 pathset-stageb.py [--passed-share X]

For every pair of NAMED_PAIRS (pathset-params.yml):
  - all R0 candidates (search at MU = inf, car included) with their fate under
    the default rules: PATHSET_DIR/stage-b/trace-<o>-<d>.parquet (contract
    TRACE of pathset/schemas.py);
  - the variants of pathset.view.variants on the same candidates: default;
    October order; R2 on weighted time; R1 keyed on the mode class; R1 keyed on
    the route pattern (corridor merge off); the other two forms of the R2 gate
    (gate_path, gate_pooled); path frequency replaced by the
    least freq_fixed / least freq_span of the links or the first link's freq;
    the deck's top 5 per (hub OD pair, mode sequence); the sensitivity values
    of EPS_T_ABS, MU and RHO_F (STAGE_B_* keys, else view.SENSITIVITY).
The R0 clause PASSED_SHARE is --passed-share, else the yml value, else the
value of the latest full run in ps-manifest (view.choose_passed_share), so
that the counts match ps-funnel and ps-trace; it is written to the report and
to the manifest. The numba kernels are compiled before the first pair is
timed (an enumeration towards the origin itself, which has no destination).
Outputs: PATHSET_DIR/stage-b/variants.parquet (one row per pair and variant)
and PATHSET_DIR/stage-b/stage-b-report.md.
"""

#%% Imports
import argparse
import os
import time

import numpy as np
import pandas as pd

from pathset import io, search
from pathset import view as V

OUT_DIR = "stage-b"
REPORT = f"{OUT_DIR}/stage-b-report.md"
LIST_MAX = 10          # routes listed per variant in the report (new / lost)
REMOVED_PER_RULE = 15  # removed alternatives listed per rule in the report
# parameters printed in the report, in this order
SHOWN_PARAMS = ["MU", "EPS_T_ABS", "EPS_T_REL", "EPS_M", "RHO_F", "K_CAP", "K_EXTRA",
                "R1_KEY", "R2_TIME", "R2_GATE", "CROSS_CLASS", "R4_GUARANTEE",
                "MAX_TRANSFERS",
                "MODE_WORDS", "T_MAX_MIN", "MIN_ACTIVE_DATES", "INCLUDE_CAR",
                "BETA_HWT", "HWT_MAX", "TAU_TP", "BETA_TRANS", "WINDOWS"]
NETWORK_STEPS = ["network", "connectors", "transfers", "ftrans", "enumerate"]


#%% Command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage B: named pairs and variants")
    ap.add_argument("--passed-share", type=float, default=None,
                    help="PASSED_SHARE (default: yml, else the latest full run)")
    return ap.parse_args(argv)


#%% One pair
def run_pair(net, o: int, d: int, params: dict, links: pd.DataFrame) -> dict:
    """Candidates, default trace (saved), every variant of one pair."""
    t0 = time.perf_counter()
    cands = V.pair_candidates(net, o, d, params, links)
    t_enum = time.perf_counter() - t0
    out = {"o": o, "d": d, "cands": cands, "t_enum": t_enum, "variants": {}}
    for name in V.variants(params):
        t = time.perf_counter()
        alts, trace = V.decide_variant(cands, params, name)
        out["variants"][name] = (alts, trace, time.perf_counter() - t)
    alts, trace, _ = out["variants"]["default"]
    io.save(V.trace_rows(cands, trace), f"{OUT_DIR}/trace-{o}-{d}")
    out["t_total"] = time.perf_counter() - t0
    return out


def variant_rows(res: dict, names, params: dict) -> pd.DataFrame:
    """One row per variant: counts, sequences, routes and the overlap with the
    default set (view.compare)."""
    cands = res["cands"].set_index("cand", drop=False)
    base = res["variants"]["default"][0]
    rows = []
    for name, (alts, trace, sec) in res["variants"].items():
        spec = V.variant_spec(name, params)
        reps = cands.loc[alts["alt"].to_numpy(np.int64)]
        ov = V.compare(base, alts)
        rows.append({
            "o": res["o"], "d": res["d"], "variant": name,
            "order": spec.get("order", "r3_first"), "freq_def": spec.get("freq", "path"),
            "changes": ";".join(f"{k}={v}" for k, v in spec.get("params", {}).items()),
            "passed_share": float(params["PASSED_SHARE"]),
            "n_cand": len(res["cands"]), "n_alt": len(alts),
            "n_capped": int((trace["status"].astype(str) == "capped").sum()),
            "n_classes": int(alts["cls"].nunique()) if len(alts) else 0,
            "n_seq": int(alts["seq"].nunique()) if len(alts) else 0,
            "seqs": V.seq_summary(alts),
            "routes": " | ".join(f"{s}: {V.route(r, names)}"
                                 for s, (_, r) in zip(alts["seq"], reps.iterrows())),
            "keys": " ".join(alts["key"].astype(str)),
            "n_base": len(base), **ov, "seconds": round(sec, 3)})
    return pd.DataFrame(rows)


#%% Report
def rule_variant_names(res: dict) -> list:
    """The variants that apply rules: all but default and deck_top5."""
    return [n for n in res["variants"] if n not in ("default", "deck_top5")]


def kept_in(res: dict) -> np.ndarray:
    """For every default alternative: the number of rule variants whose kept
    alternatives contain its representative as a member."""
    base = res["variants"]["default"][0]
    sets = [V.members(res["variants"][n][0]) for n in rule_variant_names(res)]
    return np.array([sum(int(x) in s for s in sets) for x in base["alt"]], dtype=int)


def diff_lines(res: dict, name: str, names, params: dict) -> list:
    """Routes of the variant's alternatives that no default alternative holds
    (new), and of the default alternatives the variant does not hold (lost).
    T is clock minutes; gtc and freq are those the variant used (its
    frequency, ``view.with_frequency``) for new routes and the default ones
    for lost routes."""
    spec = V.variant_spec(name, params)
    seen = V.with_frequency(res["cands"], {**params, **spec.get("params", {})},
                            spec.get("freq", "path")).set_index("cand", drop=False)
    dflt = res["cands"].set_index("cand", drop=False)
    base = res["variants"]["default"][0]
    alts = res["variants"][name][0]
    new = [a for a in alts["alt"].astype(int) if a not in V.members(base)]
    lost = [a for a in base["alt"].astype(int) if a not in V.members(alts)]
    lines = []
    for label, ids, c in (("new", new, seen), ("lost", lost, dflt)):
        if not ids:
            continue
        lines.append(f"  - {label} ({len(ids)}): " + "; ".join(
            f"{c.loc[i, 'seq']} {V.route(c.loc[i], names)} "
            f"(T {c.loc[i, 'T']:.0f}, gtc {c.loc[i, 'gtc']:.0f}, freq {c.loc[i, 'freq']:g})"
            for i in ids[:LIST_MAX]) + (" ..." if len(ids) > LIST_MAX else ""))
    return lines


TABLE_CAPTION = (
    "t_a, t_dep, t_ivt, t_trans, t_tp, t_arr and t_e are weighted minutes (deck "
    "slide 229: access and egress x BETA_ACCESS/EGRESS, riding x BETA_IVT of the "
    "mode, transfer minutes x BETA_TRANS, TAU_TP per transfer, TAU_DEP and TAU_ARR "
    "of the first and last mode) and sum to gtt; T is clock minutes door to door "
    "(the same terms without weights); hwt = gtc - gtt is the hidden wait "
    "min(BETA_HWT / (2 freq), HWT_MAX); freq is the path's departures per day "
    "(median over its dates). rides: riding minutes of each link (the mean of its "
    "riding-time class); key: the link ids. Rows with the same route use "
    "different riding-time classes (different rides and key). Their T can lie "
    "within eps_T of each other: the band of R1 is measured from the group's "
    "leader (its fastest member in the sweep), and the representative is the "
    "fastest member with freq >= RHO_F x the group's largest, which can be a "
    "slower member. in_variants: of the {n} rule variants, how many keep the "
    "representative as a member of an alternative.")


def pair_section(res: dict, table: pd.DataFrame, fuas, names, params: dict) -> str:
    o, d = res["o"], res["d"]
    alts, trace, _ = res["variants"]["default"]
    tab = V.alternatives_table(res["cands"], alts, names)
    tab["in_variants"] = kept_in(res)
    steps = V.funnel(trace, alts, "r3_first")
    lines = [f"## {V.fua_name(fuas, o)} -> {V.fua_name(fuas, d)} ({o} -> {d})", "",
             "Funnel under the default rules: "
             + " -> ".join(f"{s} {n:,}" for s, n in steps)
             + f". Enumeration {res['t_enum']:.1f} s (kernels compiled before).", "",
             "Alternatives under the default rules. "
             + TABLE_CAPTION.format(n=len(rule_variant_names(res))), "",
             V.md_table(tab[["rank", "seq", "route", "rides"] + V.COMPONENTS
                            + ["T", "gtt", "hwt", "gtc", "freq", "freq_pooled",
                               "n_variants", "in_variants", "status", "key"]]), ""]
    rem = V.removed_table(res["cands"], trace, names)
    if len(rem):
        head = V.removed_head(rem, REMOVED_PER_RULE)
        shown = ("" if len(head) == len(rem) else
                 f"; listed: the first {REMOVED_PER_RULE} of each rule in increasing "
                 "gtc (for R5 this is the cap rank)")
        lines += [f"Removed after R1 ({len(rem)}: {V.rule_counts(rem)}{shown}). "
                  "detail: the numbers of the test (R2: transfers n, time, and freq "
                  "of the remover >= RHO_F x the frequency of the removed "
                  "alternative: the largest member frequency of its group under "
                  "R2_GATE top, its representative's path frequency under path, "
                  "the pooled frequency of its group under pooled; "
                  "R5: rank in gtc among the alternatives left after R2, and the cap "
                  "max(K_CAP, classes present + K_EXTRA); a lower rank can be dropped "
                  "while a higher one stays when the higher one is the last of its "
                  "mode class, R4):", "",
                  V.md_table(head[["seq", "route", "T", "gtc", "freq", "rule",
                                   "detail"]]), ""]
    cmp_ = table[["variant", "n_alt", "n_classes", "n_same", "n_base_found",
                  "n_other_found", "seqs"]]
    lines += ["Variants (n_same: same representative as a default alternative; "
              "n_base_found: default alternatives whose representative the variant "
              "keeps as a member; n_other_found: variant alternatives whose "
              "representative is a member of a default alternative). Below, new "
              "routes carry the gtc and freq the variant used, lost routes the "
              "default ones:", "",
              V.md_table(cmp_), ""]
    for name in rule_variant_names(res):
        dl = diff_lines(res, name, names, params)
        if dl:
            lines += [f"- {name}:"] + dl
    lines.append("")
    return "\n".join(lines)


def params_section(params: dict, source: str, records: list) -> str:
    """Parameter values of the run, the R0 clause and the network inputs."""
    vals = [(k, str(params.get(k))) for k in SHOWN_PARAMS]     # str: no rounding
    vals.append(("PASSED_SHARE", f"{params['PASSED_SHARE']:g} ({source})"))
    vals.append(("DECK_TOP_K", str(params.get("DECK_TOP_K")
                                   or f"{V.DECK_TOP_K} (deck slide 238)")))
    for name, v in V.sensitivity_values(params).items():
        vals.append((V.SENSITIVITY[name][0], ", ".join(map(str, v))))
    missing = V.missing_parameters(params)
    lines = ["## Parameters and inputs", "",
             "Values used (pathset-params.yml unless stated):", "",
             V.md_table(pd.DataFrame(vals, columns=["parameter", "value"])), ""]
    if missing:
        lines += [f"Not in pathset-params.yml, read with a fallback: "
                  f"{', '.join(missing)}.", ""]
    steps = []
    for step in NETWORK_STEPS:
        rec = V.last_record(records, step)
        if rec is not None:
            info = rec.get("info", {})
            extra = (f"; MU {info.get('mu')}, PASSED_SHARE {info.get('passed_share', 'not recorded')}"
                     if step == "enumerate" else "")
            steps.append((step, rec["time"] + extra))
    files = io.fingerprint([io.OUT / f"{n}.parquet" for n in search.INPUTS]
                           + [io.out_path("ps-event-masks.npy")])
    rows = [(os.path.basename(p), f"{s / 2 ** 20:.1f}",
             time.strftime("%Y-%m-%d %H:%M", time.localtime(t))) for p, (s, t) in files.items()]
    lines += ["Latest manifest record of each step (the full run is 'enumerate'):", "",
              V.md_table(pd.DataFrame(steps, columns=["step", "time"])), "",
              "Network tables read:", "",
              V.md_table(pd.DataFrame(rows, columns=["file", "MB", "modified"])), ""]
    return "\n".join(lines)


GENERAL_NOTES = [
    "Every variant runs on the same candidates. The frequency variants change "
    "freq, hwt and gtc, and so R3, the representative of an R1 group, the gate of "
    "R2 and the order of the cap; T and gtt stay as they are.",
    "The gate of R2 (R2_GATE, printed above). Under 'pooled', the plan's wording, "
    "it compares the remover's freq with RHO_F x the POOLED frequency of the "
    "removed group (distinct first links over all its variants). Among the "
    "alternatives that enter R2 on these ten pairs the pooled frequency is a "
    "median 4 times (p90 20 times) the representative's own: the groups collect "
    "feeders from many origin stations that converge on a few trunk trains. R2 "
    "then removes about half of them (1,065 of 2,116) and the cap R5 drops 948 of "
    "the rest. Example: Helsinki -> Tallinn keeps RAB 'Pasila > Turku ~ TKU > RIX "
    "~ Riga > Tallinn' (T 967) beside BRA (T 632) of the same class, because the "
    "faster paths run once a day and the RAB group has a pooled frequency of 3 (1 "
    "< 0.5 x 3). Under 'top' (default) the gate compares with the largest path "
    "frequency among the members of the removed group, so no path is removed by "
    "one that runs less than RHO_F times as often, while a group's feeders no "
    "longer add up. Under 'path' it compares with the representative's frequency, "
    "which can be RHO_F times the group's largest, so a remover may run 1/RHO_F^2 "
    "times less often than a member it removes. The variants gate_path and "
    "gate_pooled show the other two forms. Which one to keep is an open question "
    "for Oded and Rajat (plan, 'Open with Oded and Rajat': the gate).",
    "The counts depend on PASSED_SHARE, the R0 clause that removes a path whose "
    "link passes through the origin, the destination or a city of the path on at "
    "least that share of its runs. It is printed above; with it equal to the "
    "value of the full run, the funnel of a pair equals its ps-funnel row.",
]


def totals_section(allrows: pd.DataFrame, seconds: float, n_pairs: int,
                   warmup: float, params: dict) -> str:
    g = allrows.groupby("variant", sort=False)
    tot = pd.DataFrame({
        "pairs_differing": g.apply(lambda x: int(((x["n_same"] != x["n_base"])
                                                  | (x["n_alt"] != x["n_base"])).sum()),
                                   include_groups=False),
        "alternatives": g["n_alt"].sum(), "capped": g["n_capped"].sum(),
        "pairs_capped": g["n_capped"].apply(lambda x: int((x > 0).sum())),
        "same": g["n_same"].sum(),
        "default_found": g["n_base_found"].sum(), "found_in_default": g["n_other_found"].sum(),
        "min_per_pair": g["n_alt"].min(), "max_per_pair": g["n_alt"].max(),
        "seconds": g["seconds"].sum().round(1)}).reset_index()
    n_cand = int(allrows.loc[allrows["variant"] == "default", "n_cand"].sum())
    lines = ["# Stage B: named pairs, rules and their variants", "",
             f"Pairs: {n_pairs}; R0 candidates (MU = inf, car included, PASSED_SHARE "
             f"{params['PASSED_SHARE']:g}): {n_cand:,}; run time {seconds:.0f} s, of "
             f"which {warmup:.0f} s compiling the search kernels. Generated by "
             f"pathset-stageb.py ({time.strftime('%Y-%m-%d %H:%M')}).", "",
             "## Totals over all pairs", "",
             "same: alternatives with the same representative as a default one; "
             "default_found: default alternatives whose representative the variant "
             "keeps (as representative or member); found_in_default: the reverse; "
             "pairs_differing: pairs where the set differs from the default; "
             "capped: alternatives dropped by the cap R5 (max(K_CAP, mode classes "
             "present + K_EXTRA)), pairs_capped: pairs where the cap dropped any. "
             "Where the cap acts, the number of alternatives is set by the cap and "
             "the variants differ in which alternatives are kept, not in how many. "
             "seconds: time of the rules (not the enumeration).", "",
             V.md_table(tot), "", "## What each variant changes", ""]
    lines += [f"- **{k}**: {v}" for k, v in V.variant_notes(params).items()]
    lines += [""] + [f"{t}\n" for t in GENERAL_NOTES]
    return "\n".join(lines)


def write_text(text: str, name: str) -> None:
    """Write a text file inside PATHSET_DIR atomically (io.out_path guard)."""
    path = io.out_path(name)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    io.log(f"saved {name}")


def warm_up(net, params: dict) -> float:
    """Compile the search kernels: an enumeration from the first named origin
    towards itself (no destination; search.Shard.want). Returns seconds."""
    t = time.perf_counter()
    o = int(params["NAMED_PAIRS"][0][0])
    search.candidates(net, o, [o], params)
    return time.perf_counter() - t


#%% Run
def run(args: argparse.Namespace) -> dict:
    records = V.manifest_records()
    share, source = V.choose_passed_share(args.passed_share, io.P, records)
    params = {**io.P, "PASSED_SHARE": share}
    started = time.perf_counter()
    net = search.load_network(params)
    links = io.load_out("ps-links", ["link", "freq", "freq_fixed", "freq_span", "ride"])
    names = V.hub_names(io.load_out("ps-hubs", ["hub", "kind", "name", "iata"]))
    fuas = V.load_fuas()
    t_net = time.perf_counter() - started
    warmup = warm_up(net, params)
    io.log(f"stage B: PASSED_SHARE {share:g} ({source}); network {t_net:.1f} s, "
           f"kernels {warmup:.1f} s")
    sections, tables, timing = [], [], {}
    for o, d in params["NAMED_PAIRS"]:
        res = run_pair(net, int(o), int(d), params, links)
        tab = variant_rows(res, names, params)
        tables.append(tab)
        sections.append(pair_section(res, tab, fuas, names, params))
        dv = tab.set_index("variant")["n_alt"]
        timing[f"{o}-{d}"] = round(res["t_enum"], 1)
        io.log(f"{o}->{d}: {len(res['cands']):,} candidates, default {dv['default']}, "
               f"top5 {dv['deck_top5']}, enumeration {res['t_enum']:.1f} s, "
               f"total {res['t_total']:.1f} s")
        del res
    allrows = pd.concat(tables, ignore_index=True)
    io.save(allrows, f"{OUT_DIR}/variants")
    seconds = time.perf_counter() - started
    write_text(totals_section(allrows, seconds, len(tables), warmup, params) + "\n"
               + params_section(params, source, records) + "\n"
               + "\n".join(sections), REPORT)
    info = {"pairs": len(tables), "seconds": round(seconds, 1),
            "network_s": round(t_net, 1), "compile_s": round(warmup, 1),
            "passed_share": share, "passed_share_source": source,
            "enumeration_s": timing,
            "missing_params": V.missing_parameters(io.P),
            "candidates": int(allrows.loc[allrows["variant"] == "default", "n_cand"].sum()),
            "alternatives": allrows.groupby("variant", sort=False)["n_alt"].sum().to_dict()}
    io.write_manifest("stage-b", [io.OUT / f"{n}.parquet" for n in search.INPUTS], info)
    io.log(f"stage B: {info}")
    return info


if __name__ == "__main__":
    run(parse_args())
