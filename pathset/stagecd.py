"""Stages C and D of the path enumeration: run time, memory and candidate
counts of the search, the full path set, and the sensitivity of the rules on
two origins (plan, step 6, "Staging"; Verification 4).

Plan, "Staging": stage C = "about ten origins (Paris, Ruhrgebiet, Frankfurt,
Amsterdam, Katowice, Madrid ...): run time, memory, candidate counts by mode
sequence against the deck's Table 1 (slide 223); cost of date-by-date chain
counting; sensitivity to eps_T, mu, rho_F"; stage D = "all 384 origins: full
path set". Verification 4: "384 manifest rows; pairs without alternatives in
ps-unserved; set sizes against the literature (2-3 observed, 3-15
generated)".

Sources. Everything is read from PATHSET_DIR: the combined ps-* tables of
``pathset-enumerate.py --combine``, the per-origin funnel shards (their pandas
attrs hold the search log of the origin, ``search.run_origin``), the network
tables and ps-manifest.jsonl. The console log of the stage D run is optional
(``parse_log``); without it the wall clock comes from the manifest. Only
``run_sensitivity`` searches again.

Layout. ``*_table`` and ``*_summary`` functions are pure functions of
DataFrames and dicts (tests/test_stagecd.py); ``load_*`` read files;
``section_*`` return the markdown of one section of the report;
``build_report`` joins them.

Constants of the report (the deck's Table 1, the well-known pairs, the
sensitivity origins and variants) live here, not in pathset-params.yml: a new
key in that file changes the run hash of pathset-enumerate.py (it hashes every
parameter but a short exempt list) and makes every stored shard look stale.

Sensitivity (``run_sensitivity``). One search per origin at MU = 1.0
(``SENS_SEARCH_MU``); per destination the candidates are numbered as
``search.decide`` numbers them (``view.number``), the link frequencies of the
frequency variants are added (``view.link_freqs``) and every variant of
``SENS_VARIANTS`` runs on the same candidates (``view.decide_variant``).

Why one search at MU = 1.0 serves MU 0.25 and 0.5 (checked in rules.py and
search.py). R3 of ``rules.apply_rules`` (order r3_first, ``rules.r3_cut``)
cuts a candidate iff gtt > (1 + MU) x the least gtt of its mode class AND
gtc > (1 + MU) x the least gtc of its class, both minima over the candidates
passed in. The search's R3 is the same test with the minima over all R0
candidates of the pair (search.py, "R3"; pass 1 computes the minima exactly,
values compared as float32, as apply_rules sees them). The candidates that
hold the two minima of a class are never cut, so they are among the MU = 1.0
candidates; the minima over that set equal those over all R0 candidates; a
candidate cut at MU = 1.0 is cut at every smaller MU. Hence R3 at MU <= 1 on
the MU = 1.0 set keeps exactly what R3 at that MU keeps among all R0
candidates, and everything after R3 (the band of R1 uses the least T of the
candidates left) is the same. This needs gtt and gtc as the search computed
them. The frequency variants change gtc; for them the result is exact for a
pair when every class satisfies ``r3_exact`` (a candidate outside the set has
gtt > 2 x the class minimum of gtt, so gtc' >= gtt is beyond the margin), and
the report counts the pairs where this is not guaranteed. The October order
and the deck's top 5 need every R0 candidate (as in stage B) and are left out.
The rules work per pair, so ``sensitivity_pair`` runs every variant on the
candidates of one pair and ``run_sensitivity`` adds up the pairs of an
origin. sensitivity-runs keeps, per origin, the hash of the parameters and
inputs (``run_hash``), a hash of the code that made the tables
(``sens_code_hash``) and the memory of the process; the report compares
them with the present ones.

Section 11 also lists what the review of the first version of this report
found in the data and the design (items 12 to 23). Their numbers are
computed here from the stored tables; three of them read the network
inputs (data-upstream-hubs stations and timetables, data-upstream flights)
and say so when those files are missing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import math
import platform
import re
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import dates as D
from . import io
from . import rules
from . import search
from . import view as V

# --- Outputs (inside PATHSET_DIR) --------------------------------------------
OUT_DIR = "stage-cd"
REPORT = f"{OUT_DIR}/stage-cd-report.md"
LOG_COPY = f"{OUT_DIR}/stage-d-console.log"     # copy of the console log of stage D
SENS_TABLE = f"{OUT_DIR}/sensitivity"           # one row per origin x variant
SENS_ALTS = f"{OUT_DIR}/sensitivity-alts"       # every alternative of every variant
SENS_RUNS = f"{OUT_DIR}/sensitivity-runs"       # one row per origin: search and checks
SENS_MISMATCH = f"{OUT_DIR}/sensitivity-default-mismatch"

# --- The deck's Table 1 (slide 223) ------------------------------------------
DECK_FILE = "RA/Nov/path enumeration/3MARS Meetings - RV.pptx"
DECK_TABLE1_COLS = ["category", "seq", "paths", "acc_mean", "acc_sd", "egr_mean",
                    "egr_sd", "ride_A_mean", "ride_A_sd", "ride_B_mean", "ride_B_sd",
                    "ride_R_mean", "ride_R_sd", "km_mean", "km_sd"]
# "Table 1: Path counts and mean (standard deviation) by mode sequence":
# number of paths, access and egress time (min), IVT (min) of air, bus and
# rail, length (km). Read from the picture of slide 223.
DECK_TABLE1 = pd.DataFrame([
    ("Direct", "A", 492_124, 97, 47, 97, 47, 115, 69, 0, 0, 0, 0, 1197, 493),
    ("Direct", "B", 20_095, 25, 24, 25, 24, 0, 0, 566, 428, 0, 0, 667, 459),
    ("Direct", "R", 11_199, 22, 15, 22, 15, 0, 0, 0, 0, 232, 186, 426, 291),
    ("A2A", "AA", 42_853_247, 97, 46, 97, 46, 264, 88, 0, 0, 0, 0, 2662, 950),
    ("Air1", "AB", 146_002, 98, 47, 18, 6, 109, 38, 498, 328, 0, 0, 1618, 674),
    ("Air1", "AR", 19_144, 99, 48, 15, 5, 98, 34, 0, 0, 100, 73, 959, 461),
    ("Air1", "BA", 146_326, 18, 6, 98, 47, 131, 171, 493, 325, 0, 0, 1615, 672),
    ("Air1", "RA", 19_648, 14, 5, 99, 48, 143, 258, 0, 0, 101, 74, 957, 453),
    ("PT2", "BB", 1_459_256, 26, 25, 26, 25, 0, 0, 1248, 688, 0, 0, 1414, 727),
    ("PT2", "BR", 434_813, 27, 26, 23, 15, 0, 0, 532, 384, 234, 176, 1011, 511),
    ("PT2", "RB", 434_445, 23, 14, 27, 26, 0, 0, 531, 381, 237, 181, 1014, 515),
    ("PT2", "RR", 440_034, 22, 13, 22, 13, 0, 0, 0, 0, 504, 291, 876, 434),
], columns=DECK_TABLE1_COLS)
# What the deck says around Table 1 (slide text, quoted)
DECK_QUOTES = {
    222: ["523k direct & 45.95M single-transfer paths computed between 146k city pairs",
          "Computing 2-transfer paths infeasible (count blows up exponentially)"],
    232: ["Service link in 3MG = (source hub, target hub, mode, operator)",
          "(5 random days per month in [2023, 2025])",
          "4.72M service events (rows) across 180 days with 34.6k service links"],
    238: ["Top 5 paths with the least (mode-weighted) IVT (no transfer time) for each "
          "combination of (city pair, hub OD pair, mode sequence)", "33.4M paths",
          "35 mode sequences with 0, 1 & 2 transfers",
          ">375M paths without the above filtering"],
}
PLAN_LITERATURE = "2-3 observed, 3-15 generated"     # plan, Verification 4

# --- Well-known pairs (section 9) --------------------------------------------
WELL_KNOWN_PAIRS = [
    ("Paris", "London"), ("London", "Paris"), ("Paris", "Lyon"),
    ("Paris", "Bruxelles"), ("Amsterdam", "Bruxelles"), ("Amsterdam", "Berlin"),
    ("Amsterdam", "Muenchen"), ("Frankfurt", "Koeln"), ("Berlin", "Hamburg"),
    ("Berlin", "Warszawa"), ("Muenchen", "Wien"), ("Wien", "Budapest"),
    ("Praha", "Wien"), ("Zuerich", "Milano"), ("Milano", "Roma"), ("Roma", "Napoli"),
    ("Madrid", "Barcelona"), ("Madrid", "Sevilla"), ("Lisboa", "Porto"),
    ("Koebenhavn", "Stockholm"), ("Stockholm", "Oslo"), ("Helsinki", "Tallinn"),
    ("Warszawa", "Krakow"), ("Dublin", "London"), ("Athina", "Thessaloniki")]
# Tried when a name does not resolve (pathset.view.resolve_fua folds accents
# but not the German and Danish transliterations)
LOCAL_NAMES = {"Muenchen": "München", "Koeln": "Köln", "Zuerich": "Zürich",
               "Koebenhavn": "København"}

# --- Sensitivity (section 10) ------------------------------------------------
SENS_ORIGINS = (367, 148)            # Wien, Katowice
SENS_SEARCH_MU = 1.0                 # one search per origin; see the docstring
SENS_VARIANTS = {
    "default": {},
    "eps_10": {"params": {"EPS_T_ABS": 10}},
    "eps_20": {"params": {"EPS_T_ABS": 20}},
    "eps_rel_0": {"params": {"EPS_T_REL": 0.0}},
    "mu_0.25": {"params": {"MU": 0.25}},
    "mu_1.0": {"params": {"MU": 1.0}},
    "rho_0.33": {"params": {"RHO_F": 0.33}},
    "rho_1.0": {"params": {"RHO_F": 1.0}},
    "gate_path": {"params": {"R2_GATE": "path"}},
    "gate_pooled": {"params": {"R2_GATE": "pooled"}},
    "r1_class": {"params": {"R1_KEY": "class"}},
    "r1_pattern": {"params": {"R1_KEY": "pattern"}},
    "r2_weighted": {"params": {"R2_TIME": "weighted"}},
    "cross_class": {"params": {"CROSS_CLASS": True}},
    "cross_class_no_r4": {"params": {"CROSS_CLASS": True, "R4_GUARANTEE": False}},
    "no_cap": {"params": {"K_CAP": 1_000_000}},
    "freq_fixed": {"freq": "fixed"},
    "freq_span": {"freq": "span"},
    "freq_first": {"freq": "first"},
}
SENS_NOTES = {
    "default": "the rules of the run (pathset-params.yml)",
    "eps_10": "EPS_T_ABS 10 min instead of 15",
    "eps_20": "EPS_T_ABS 20 min instead of 15",
    "eps_rel_0": ("EPS_T_REL 0: a flat 15-minute band, no 5% term (plan, open: flat "
                  "15 min or the 5% term)"),
    "mu_0.25": "R3 margin MU 0.25 instead of 0.5",
    "mu_1.0": "R3 margin MU 1.0 instead of 0.5",
    "rho_0.33": "frequency gate RHO_F 0.33 instead of 0.5",
    "rho_1.0": "frequency gate RHO_F 1.0 instead of 0.5",
    "gate_path": ("R2_GATE path: the remover must run at least RHO_F times as often as "
                  "the representative of the removed group"),
    "gate_pooled": ("R2_GATE pooled (the plan's wording): ... as the pooled frequency of "
                    "the removed group (distinct first links)"),
    "r1_class": "R1 keyed on mode class and transfer count (RB and BR near-ties merge)",
    "r1_pattern": ("R1 keyed on mode sequence and route pattern (near-ties through "
                   "different transfer cities stay apart)"),
    "r2_weighted": "R2 compares gtt (weighted) instead of the clock time T",
    "cross_class": ("CROSS_CLASS on: strict R2 across mode classes on the survivors; "
                    "R4 gives a class back its best survivor"),
    "cross_class_no_r4": ("CROSS_CLASS on and R4_GUARANTEE off: a class can lose all its "
                          "alternatives to the comparison across classes (the cap still "
                          "never drops the last of a class, rules.cap_drop)"),
    "no_cap": "K_CAP 1,000,000: no cap, the set before R5",
    "freq_fixed": ("path frequency = least freq_fixed of the links (2023-2025 "
                   "departures / days, the deck's fixed denominator, slide 237); it is 0 "
                   "for every link whose data start in 2026, so this variant mostly "
                   "measures the feed dates (lessons below)"),
    "freq_span": "path frequency = least freq_span of the links (first to last date)",
    "freq_first": "path frequency = freq of the first link (the deck's hidden-wait form)",
}
SENS_LEFT_OUT = {
    "october": "R1 and R2 run on all R0 candidates before the margin; it needs every R0 "
               "candidate, as in stage B",
    "deck_top5": "the deck's top 5 per (hub OD pair, mode sequence) by t_ivt over all R0 "
                 "candidates; it needs every R0 candidate, as in stage B",
    "abs_gate": "the thesis's absolute gate, under which only a path with a headway of at "
                "most 15 min on every leg may remove another (Cats 2011 eq. 5.5; plan, 'Open "
                "with Oded and Rajat'); rules.gate has only the relative form, and this "
                "report does not change the rules code",
}
FREQ_VARIANTS = ("freq_fixed", "freq_span", "freq_first")
# Refuse an origin whose search at MU = 1.0 would be too large for memory:
# the candidates at MU = 1.0 were 3.3 times ps-funnel n_r3 (MU = 0.5) for
# Wien (4.19 million); Ruhrgebiet has 6.48 million at MU = 0.5
SENS_MAX_R3 = 3_000_000
# Code that makes the sensitivity tables (sens_code_hash): the files the
# search runs, the viewer's variant code and these functions of this module
SENS_FILES = ["pathset/view.py"]
SENS_FUNCS = ("sensitivity_specs", "sensitivity_pair", "run_sensitivity", "r3_exact",
              "r2_details", "oded_removals", "overlap", "default_mismatch")

# --- Open items of the night of 13-14 Nov (section 11) -----------------------
NIGHT_T_BASIS = {7: (1380, 1970), 133: (1386, 1846)}   # (disjoint, T-basis), as noted
NIGHT_PASS_THROUGH_NAMED = 29
NIGHT_EPS_MAX = 48.7
NIGHT_ODED_STAGE_B = 4
N_NIGHT_ITEMS = 11

# --- What the review of this report found (section 11, items 12 to 23) -------
REVIEW_ITEMS = ("eurostar", "fast_links", "holes", "carriers", "class_split",
                "near_ties", "distance", "car", "sea_access", "stations", "tz_coach",
                "rare")


def item_no(name: str) -> int:
    """Number of a review item in section 11 (after the night's items)."""
    return N_NIGHT_ITEMS + 1 + REVIEW_ITEMS.index(name)


# Links faster than any real service (review, 14 Nov). Each entry: link id ->
# (board hub, alight hub, riding minutes, what is wrong). The report uses an
# entry only when ps-links still has that link with these hubs and riding
# time (``suspect_links``), so a rebuilt network cannot inherit a stale flag.
SUSPECT_LINKS = {
    5420: ("Paris Gare du Nord", "St-Pancras-International", 121.0,
           "a dwell of 18 min at the first call in feed 5 (man-EuroStar)"),
    5158: ("St-Pancras-International", "Paris Gare du Nord", 121.0,
           "the same feed; other runs take 138 min or more"),
    28679: ("Lyon Part Dieu", "Paris Gare de Lyon Hall 1 - 2", 91.3,
            "a dwell of 24 to 33 min at Lyon in four feeds"),
    17989: ("Padova", "Vicenza", 6.0, "30 km in 6 min"),
    25246: ("Karlsruhe Hauptbahnhof", "Mannheim, Hauptbahnhof", 13.2, "54 km in 13 min"),
    37608: ("Brno dolní nádraží", "Praha Florenc/Busbahnhof", 90.0,
            "coach, 186 km straight line in 90 min"),
    14603: ("ZOB", "Rosenheim", 30.0, "coach, 53 km straight line in 30 min"),
    62442: ("Lausanne (P+R Vélodrome)", "St. Gallen", 120.0,
            "coach, 230 km straight line in 120 min"),
    42308: ("Trieste (Autostazione)", "Verona Porta Nuova", 120.0,
            "coach, 219 km straight line in 120 min"),
    17806: ("Verona Porta Nuova", "Trieste (Autostazione)", 120.0,
            "coach, 219 km straight line in 120 min"),
}
# Plan, Verification 2: "known data holes reported, not failed"
PLAN_KNOWN_HOLES = {
    ("Amsterdam", "Berlin"): "Intercity on 11 dates",
    ("Madrid", "Barcelona"): "no non-stop train",
    ("Praha", "Wien"): "rail about 1 a day",
    ("Zuerich", "Milano"): "no rail",
}
# Data holes on the well-known pairs that the plan's list misses (review, 14
# Nov; evidence in item "holes" or the item named)
REVIEW_HOLES = {
    ("Paris", "London"): ("the 121-min class of Paris Nord > St-Pancras is a feed error",
                          "eurostar"),
    ("London", "Paris"): ("the 121-min class of St-Pancras > Paris Nord is a feed error",
                          "eurostar"),
    ("Berlin", "Warszawa"): ("direct trains only on 3 dates (ps-sparse)", "holes"),
    ("Muenchen", "Wien"): ("few direct trains between the main stations", "holes"),
    ("Wien", "Budapest"): ("direct trains on 31 dates, 1 a day", "holes"),
    ("Madrid", "Barcelona"): ("high-speed trains 1 to 3 a day per class, from late June 2026",
                              "holes"),
    ("Madrid", "Sevilla"): ("one rail link, 3 a day, and no coach", "holes"),
    ("Lisboa", "Porto"): ("Porto's rail station in the data is Espinho", "stations"),
    ("Dublin", "London"): ("Aer Lingus (EI) is not in the air data", "carriers"),
    ("Athina", "Thessaloniki"): ("rail on 19 dates in January 2025 from Acharnes; coach only "
                                 "in ps-sparse", "holes"),
}
# Pairs whose direct services item "holes" lists, with the modes looked at
HOLE_PAIRS = [("Berlin", "Warszawa", "R"), ("Muenchen", "Wien", "R"), ("Wien", "Budapest", "R"),
              ("Madrid", "Barcelona", "R"), ("Madrid", "Sevilla", "RB"),
              ("Sevilla", "Malaga", "ABR"), ("Oslo", "Bergen", "R"),
              ("Athina", "Thessaloniki", "RB"), ("Palermo", "Catania", "B"),
              ("Sofia", "Plovdiv", "B")]
# Flights in the air data of three routes where the absent carriers fly (item
# "carriers"); carriers looked for
CARRIER_ROUTES = [("PMI", "BCN"), ("MAD", "BCN"), ("DUB", "LHR")]
CARRIERS_ABSENT = {"VY": "Vueling", "EI": "Aer Lingus", "UX": "Air Europa"}
# Item "stations": FUAs whose main rail station the review found missing, and
# connectors to compare (FUA, name pattern)
STATION_FUAS = ["Porto", "Lisboa", "Warszawa", "Athina"]
CONNECTOR_CHECKS = [("London", "St-Pancras|Paddington|Euston|Heathrow Terminal [25]$"),
                    ("Liverpool", "Lime Street|Newton-le-Willows")]
DISTANCE_BANDS = [0, 150, 300, 600, 1000, 1500, 100_000]     # km between FUA centres

# Copy of pathset-enumerate.py: parameters that do not enter its run hash
ENUMERATE_UNHASHED = ("WORKERS", "MEM_CAND_GB", "STAGE_C_ORIGINS", "STAGE_B_EPS_T_ABS",
                      "STAGE_B_MU", "STAGE_B_RHO_F", "DECK_TOP_K")
# Code that produced the shards (checked against the run's start time)
RUN_CODE = ["pathset/__init__.py", "pathset/cost.py", "pathset/dates.py", "pathset/io.py",
            "pathset/rules.py", "pathset/schemas.py", "pathset/search.py",
            "pathset/transfer.py", "pathset-enumerate.py", "pathset-params.yml"]
# Files scanned for callers of pairs.ftrans_table (open item 10)
FTRANS_SCAN = ["pathset-transfers.py", "pathset-checks.py", "pathset-stageb.py",
               "pathset-view.py", "pathset-enumerate.py", "pathset-network.py",
               "pathset-connectors.py", "pathset/pairs.py", "pathset/transfer.py",
               "pathset/view.py", "pathset/search.py"]

MODES = ("A", "B", "R")
CLASSES = ("A", "AB", "ABR", "AR", "B", "BR", "R", "C")


# --- Formatting --------------------------------------------------------------
def n_(x) -> str:
    """An integer with thousands separators; '' for NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{int(round(float(x))):,}"


def f_(x, digits: int = 1) -> str:
    """A float with thousands separators and ``digits`` decimals; '' for NaN."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return ""
    if isinstance(x, (float, np.floating)) and np.isinf(x):
        return "inf"
    return f"{float(x):,.{digits}f}"


def pct(a, b, digits: int = 1) -> str:
    """a / b as a percentage; '' when b is 0."""
    return "" if not b else f"{100.0 * float(a) / float(b):.{digits}f}%"


def hm(minutes) -> str:
    """Minutes as h:mm, rounded to the minute ('' for NaN)."""
    if minutes is None or not np.isfinite(minutes):
        return ""
    m = int(round(float(minutes)))
    return f"{m // 60}:{m % 60:02d}"


def wall_text(seconds) -> str:
    """Seconds as 'H h M min', rounded to the minute (never '60 min')."""
    h, m = divmod(int(round(float(seconds) / 60)), 60)
    return f"{h} h {m} min"


def num(x) -> str:
    """A number as the deck prints it: thousands separators, no decimals
    when it is whole."""
    v = float(x)
    return f"{v:,.0f}" if v == int(v) else f"{v:,g}"


def mean_sd(values, digits: int = 0) -> str:
    """'mean (sd)' of a series, as the deck prints its Table 1; '' when empty."""
    v = pd.Series(values, dtype="float64").dropna()
    if not len(v):
        return ""
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():,.{digits}f} ({sd:,.{digits}f})"


def md(df: pd.DataFrame, digits: int = 1) -> str:
    """Markdown table: integers with thousands separators, floats with
    ``digits`` decimals and separators, NaN empty, '|' escaped."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            return f_(v, digits)
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def quantiles(values, qs=(0.5, 0.9)) -> dict:
    """Quantiles of a series (linear interpolation) and its max."""
    v = pd.Series(values, dtype="float64").dropna()
    out = {f"p{int(round(q * 100))}": float(v.quantile(q)) for q in qs}
    out["max"] = float(v.max()) if len(v) else math.nan
    return out


# --- Search log of the shards ------------------------------------------------
def search_stats(attrs: dict) -> dict:
    """Flat counters and seconds of one origin from the attrs of its funnel
    shard (``search.run_origin``: attrs["search"] with origin, blocks,
    buffer_mb, rules_s, total_s, rss_start_mb, peak_mb and per pass a dict of
    seconds ``s`` and the counters ``search.STAT_NAMES``).

    pass 1 finds the class minima, pass 2 counts and keeps the candidates,
    trace_pass2 (named pairs only) enumerates them at MU = inf. other_s =
    total - passes - rules: the exact rows, the tables and best-by-sequence.
    chains3: three-link chains that passed the prefix clauses (pass 2);
    skip_share: chains3_skipped / chains3 of the pass (not evaluated because
    their bounds lie outside what the pass needs, or no destination was left);
    calls: transfer.triple_stats evaluations (cache misses); reused: cache
    hits; evaluated: R0 candidates with exact values (pass 2); calls_per_s:
    calls of passes 1 and 2 over their seconds (the passes also walk the
    chains and compute costs, so the calls alone are faster than this).
    """
    s = attrs.get("search", attrs)
    p1, p2, tr = s.get("pass1", {}), s.get("pass2", {}), s.get("trace_pass2", {})
    out = {"origin": int(s["origin"]), "total_s": float(s["total_s"]),
           "pass1_s": float(p1.get("s", 0.0)), "pass2_s": float(p2.get("s", 0.0)),
           "trace_s": float(tr.get("s", 0.0)), "rules_s": float(s["rules_s"]),
           "blocks": int(s["blocks"]), "buffer_mb": float(s["buffer_mb"]),
           "rss_start_mb": float(s["rss_start_mb"]), "peak_mb": float(s["peak_mb"])}
    out["other_s"] = (out["total_s"] - out["pass1_s"] - out["pass2_s"] - out["trace_s"]
                      - out["rules_s"])
    for k in ("chains1", "chains2", "chains3"):
        out[k] = int(p2.get(k, 0))
    for i, p in ((1, p1), (2, p2)):
        ch = int(p.get("chains3", 0))
        out[f"skipped{i}"] = int(p.get("chains3_skipped", 0))
        out[f"skip_share{i}"] = out[f"skipped{i}"] / ch if ch else math.nan
        out[f"calls{i}"] = int(p.get("triple_calls", 0))
    out["calls_trace"] = int(tr.get("triple_calls", 0))
    out["reused2"] = int(p2.get("triple_reused", 0))
    out["evaluated2"] = int(p2.get("evaluated", 0))
    out["r0_by_bounds"] = int(p2.get("r0_by_bounds", 0))
    out["cache_max"] = max(int(p.get("cache_entries", 0)) for p in (p1, p2, tr))
    out["cache_cleared"] = sum(int(p.get("cache_cleared", 0)) for p in (p1, p2, tr))
    secs = out["pass1_s"] + out["pass2_s"]
    out["calls_per_s"] = (out["calls1"] + out["calls2"]) / secs if secs > 0 else math.nan
    return out


def timing_table(attrs_by_origin: dict) -> pd.DataFrame:
    """``search_stats`` of every origin, one row each, by origin."""
    rows = [search_stats(a) for _, a in sorted(attrs_by_origin.items())]
    return pd.DataFrame(rows)


def load_shard_attrs(folder: Path | None = None) -> dict:
    """Pandas attrs of every funnel shard (o -> {"search": ..., "param_hash":
    ...}), read from the parquet schema only."""
    folder = io.OUT / "paths" / "funnel" if folder is None else Path(folder)
    out = {}
    for path in sorted(folder.glob("o*.parquet")):
        meta = pq.read_schema(path).metadata or {}
        out[int(path.stem[1:])] = json.loads(meta.get(b"PANDAS_ATTRS", b"{}"))
    return out


# --- Console log and manifest ------------------------------------------------
_LOG_LINE = re.compile(r"^(\d\d):(\d\d):(\d\d) (.*)$")
_LOG_START = re.compile(r"^enumerate: (\d+) origins to run, (\d+) finished, (\d+) workers, "
                        r"MU ([^,]+), PASSED_SHARE ([^,]+), hash (\w+)")


def parse_log(text: str, start_date: dt.date) -> dict:
    """Start and end of a pathset-enumerate.py console log.

    Lines start with HH:MM:SS (io.log) and carry no date: the first line is
    on ``start_date`` and the date advances by one day when a time is more
    than 12 hours earlier than the one before (a run past midnight; a line
    a little out of order, as several writers can produce, does not move
    the date). start: the line
    'enumerate: N origins to run ...' (with origins, workers, MU,
    PASSED_SHARE, hash); end: the last 'saved paths/funnel/...' line (an
    origin is complete when its funnel shard is saved, pathset-enumerate.py);
    origins: lines 'o N: ...'. Returns a dict; start or end None when absent.
    """
    day, last = start_date, None
    out = {"start": None, "end": None, "origins": 0, "funnels": 0}
    for line in text.splitlines():
        m = _LOG_LINE.match(line)
        if not m:
            continue
        t = dt.time(int(m[1]), int(m[2]), int(m[3]))
        secs = t.hour * 3600 + t.minute * 60 + t.second
        if last is not None and secs < last - 12 * 3600:
            day += dt.timedelta(days=1)
        if last is None or secs < last - 12 * 3600 or secs > last:
            last = secs
        stamp, msg = dt.datetime.combine(day, t), m[4]
        s = _LOG_START.match(msg)
        if s and out["start"] is None:
            out.update(start=stamp, n_to_run=int(s[1]), workers=int(s[3]), mu=s[4],
                       passed_share=s[5], hash=s[6])
        elif msg.startswith("saved paths/funnel/"):
            out["end"] = stamp
            out["funnels"] += 1
        elif re.match(r"^o \d+: ", msg):
            out["origins"] += 1
    if out["start"] is not None and out["end"] is not None:
        out["seconds"] = (out["end"] - out["start"]).total_seconds()
    return out


def run_records(records: list) -> tuple:
    """(run, combine): the latest ps-manifest 'enumerate' record that ran
    origins, and the latest 'enumerate' record after it that combined the
    shards (None when there is none)."""
    run_i = max((i for i, r in enumerate(records) if r.get("step") == "enumerate"
                 and r.get("info", {}).get("origins_run")), default=None)
    if run_i is None:
        return None, None
    comb = [r for r in records[run_i:] if r.get("step") == "enumerate"
            and r.get("info", {}).get("combined")]
    return records[run_i], (comb[-1] if comb else None)


def record_of(records: list, step: str, origins: list | None = None) -> dict | None:
    """The latest manifest record of ``step`` (with exactly these origins run
    when given)."""
    for rec in reversed(records):
        if rec.get("step") != step:
            continue
        if origins is None or sorted(rec.get("info", {}).get("origins_run", [])) == sorted(origins):
            return rec
    return None


def run_hash(params: dict, named: bool, files: list) -> str:
    """The run hash of pathset-enumerate.py (copy of its ``run_hash``): sha256
    of the parameters but ``ENUMERATE_UNHASHED``, the --named switch and the
    size and time of the input files, first 16 hex digits."""
    keep = {k: v for k, v in params.items() if k not in ENUMERATE_UNHASHED}
    blob = json.dumps({"params": keep, "named": named, "inputs": io.fingerprint(files)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def input_files() -> list:
    """The input files of pathset-enumerate.py (its ``input_files``)."""
    return [io.OUT / f"{n}.parquet" for n in search.INPUTS] + [io.OUT / "ps-event-masks.npy"]


def git_info(root: Path = io.ROOT, paths=("pathset", "pathset-enumerate.py",
                                          "pathset-params.yml")) -> dict:
    """HEAD commit, its date and ``git status --porcelain`` of ``paths``
    (read-only git commands); empty values when git fails."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  text=True, timeout=30, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"head": run("rev-parse", "--short", "HEAD"),
            "date": run("log", "-1", "--format=%ci"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status": run("status", "--porcelain", "--", *paths)}


def file_times(root: Path, names: list) -> pd.DataFrame:
    """Modification time of each file (local time)."""
    rows = []
    for n in names:
        p = Path(root) / n
        if p.exists():
            rows.append((n, dt.datetime.fromtimestamp(p.stat().st_mtime)))
    return pd.DataFrame(rows, columns=["file", "modified"])


# --- Funnel, sequences, sets -------------------------------------------------
def r0_interval(n_r0, n_r0_bound):
    """(low, high) of the exact R0 count: n_r0 - n_r0_bound <= exact <= n_r0
    (search.py, "Counting convention"). Raises when a bound part exceeds the
    count."""
    hi = np.asarray(n_r0, dtype=np.int64)
    b = np.asarray(n_r0_bound, dtype=np.int64)
    if (b > hi).any() or (b < 0).any():
        raise ValueError("n_r0_bound must lie in [0, n_r0]")
    return hi - b, hi


def seq_table(seq_counts: pd.DataFrame) -> pd.DataFrame:
    """Counts per mode sequence over all origins (ps-seq-counts summed):
    links (0 for the car), n_r0 with its interval r0_low..r0_high, exact
    (no part counted by bounds), n_r3, n_kept. Ordered by links, then seq."""
    g = seq_counts.groupby("seq", sort=True)[["n_r0", "n_r0_bound", "n_r3", "n_kept"]].sum()
    out = g.reset_index()
    out["links"] = [0 if s == "C" else len(s) for s in out["seq"]]
    out["r0_low"], out["r0_high"] = r0_interval(out["n_r0"], out["n_r0_bound"])
    out["exact"] = out["n_r0_bound"] == 0
    return (out.sort_values(["links", "seq"], kind="stable").reset_index(drop=True)
            [["seq", "links", "n_r0", "n_r0_bound", "r0_low", "r0_high", "exact",
              "n_r3", "n_kept"]])


def _link_arrays(paths: pd.DataFrame, links: pd.DataFrame):
    lk = links.sort_values("link", kind="stable")
    if not (lk["link"].to_numpy() == np.arange(len(lk))).all():
        raise ValueError("ps-links.link must be 0 .. L-1")
    ls = np.column_stack([paths[f"l{k}"].to_numpy(np.int64) for k in (1, 2, 3)])
    used = ls >= 0
    return lk, used, np.where(used, ls, 0)


def path_metrics(paths: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Per path: acc and egr (clock minutes of the car access and egress,
    PATH columns), ride_A, ride_B, ride_R (unweighted riding minutes of the
    path's links of that mode: ps-links ride, the date-weighted mean of the
    link's riding-time class; 0 without such a link) and km (sum of ps-links
    km of the links: great-circle distance between the board and alight hub
    of each link, access and egress not included; NaN for the car)."""
    lk, used, safe = _link_arrays(paths, links)
    ride = np.where(used, lk["ride"].to_numpy(np.float64)[safe], 0.0)
    mode = lk["mode"].astype(str).to_numpy()[safe]
    out = pd.DataFrame({"seq": paths["seq"].to_numpy(),
                        "acc": paths["acc"].to_numpy(np.float64),
                        "egr": paths["egr"].to_numpy(np.float64)})
    for m in MODES:
        out[f"ride_{m}"] = (ride * (used & (mode == m))).sum(axis=1)
    km = np.where(used, lk["km"].to_numpy(np.float64)[safe], 0.0).sum(axis=1)
    out["km"] = np.where(used.any(axis=1), km, np.nan)
    return out


METRICS = ["acc", "egr", "ride_A", "ride_B", "ride_R", "km"]


def metrics_by_seq(metrics: pd.DataFrame, seqs) -> pd.DataFrame:
    """Rows and 'mean (sd)' of every metric per mode sequence (``seqs``, in
    that order)."""
    rows = []
    for s in seqs:
        x = metrics[metrics["seq"] == s]
        rows.append({"seq": s, "rows": len(x), **{k: mean_sd(x[k]) for k in METRICS}})
    return pd.DataFrame(rows)


def cap_of(n_classes, params: dict) -> np.ndarray:
    """R5 cap per pair: max(K_CAP, classes present + K_EXTRA)."""
    return np.maximum(int(params["K_CAP"]),
                      np.asarray(n_classes, dtype=np.int64) + int(params["K_EXTRA"]))


def set_size_table(funnel: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Set size per number of mode classes kept (n_classes of ps-funnel; with
    CROSS_CLASS off every class of the R3 candidates keeps an alternative, so
    it is also the number of classes present): pairs, the cap, median, mean
    and largest set size, pairs at the cap, pairs where the cap dropped any
    alternative and how many it dropped."""
    f = funnel.assign(cap=cap_of(funnel["n_classes"], params))
    g = f.groupby("n_classes", sort=True)
    out = pd.DataFrame({
        "pairs": g.size(), "cap": g["cap"].first(),
        "median": g["n_kept"].median(), "mean": g["n_kept"].mean(),
        "max": g["n_kept"].max(),
        "at_cap": g.apply(lambda x: int((x["n_kept"] == x["cap"]).sum()), include_groups=False),
        "pairs_capped": g["n_capped"].apply(lambda x: int((x > 0).sum())),
        "capped": g["n_capped"].sum()}).reset_index()
    return out


def size_distribution(funnel: pd.DataFrame) -> pd.DataFrame:
    """Pairs by number of kept alternatives."""
    n = funnel["n_kept"].value_counts().sort_index()
    return pd.DataFrame({"alternatives": n.index.astype(int), "pairs": n.to_numpy()})


def class_extras(alts: pd.DataFrame) -> dict:
    """One alternative per class against extra ones: pairs, alternatives,
    (pair, class) cells, alternatives beyond the first of their class, and per
    class the cells and the cells with more than one alternative."""
    per = alts.groupby(["o", "d", "cls"], sort=False).size()
    by_cls = per.groupby(level="cls")
    return {"pairs": int(alts.groupby(["o", "d"]).ngroups), "alternatives": len(alts),
            "cells": len(per), "extras": int((per - 1).sum()),
            "per_class": pd.DataFrame({"cells": by_cls.size(),
                                       "multi": by_cls.apply(lambda x: int((x > 1).sum())),
                                       "alternatives": by_cls.sum()}).reset_index()}


# --- Basis of the path frequency ---------------------------------------------
def common_period(paths: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Per path: first (the latest first_day of its links), last (the
    earliest last_day) and disjoint = first > last, i.e. no date lies in the
    [first_day, last_day] range of every link (ps-links; the car is never
    disjoint)."""
    lk, used, safe = _link_arrays(paths, links)
    fd = np.where(used, lk["first_day"].to_numpy(np.int64)[safe], np.iinfo(np.int64).min)
    ld = np.where(used, lk["last_day"].to_numpy(np.int64)[safe], np.iinfo(np.int64).max)
    first, last = fd.max(axis=1), ld.min(axis=1)
    any_link = used.any(axis=1)
    return pd.DataFrame({"first": np.where(any_link, first, -1),
                         "last": np.where(any_link, last, -1),
                         "disjoint": any_link & (first > last)}, index=paths.index)


def basis_table(alts: pd.DataFrame, disjoint: np.ndarray) -> pd.DataFrame:
    """Per mode class: alternatives, basis D and T, share T, and the T-basis
    alternatives whose links never ran in a common period (``disjoint``)."""
    x = alts.assign(disjoint=np.asarray(disjoint, dtype=bool),
                    is_T=(alts["basis"].astype(str) == "T").to_numpy(),
                    is_D=(alts["basis"].astype(str) == "D").to_numpy())
    g = x.groupby("cls", sort=True)
    out = pd.DataFrame({"alternatives": g.size(), "basis_D": g["is_D"].sum(),
                        "basis_T": g["is_T"].sum(),
                        "T_disjoint": g.apply(lambda t: int((t["is_T"] & t["disjoint"]).sum()),
                                              include_groups=False)}).reset_index()
    out["share_T"] = [pct(t, n) for t, n in zip(out["basis_T"], out["alternatives"])]
    out["share_T_disjoint"] = [pct(d, t) for d, t in zip(out["T_disjoint"], out["basis_T"])]
    return out


# --- Overlap of a variant with the default -----------------------------------
def cap_detail(default: pd.DataFrame, no_cap: pd.DataFrame, params: dict) -> dict:
    """What the cap R5 did on one origin, from two variants of the same
    candidates (sensitivity-alts rows: d, rank, key, cls): ``no_cap`` holds
    the set before R5 (rank in increasing gtc), ``default`` the kept set.
    dropped: classes of the alternatives the cap dropped (in no_cap, not
    kept); protected: kept alternatives whose rank before the cap is beyond
    the cap max(K_CAP, classes present + K_EXTRA), i.e. kept only because
    they are the last of their class (R4), with their classes."""
    nc = no_cap.copy()
    nc["n_cls"] = nc.groupby("d")["cls"].transform("nunique")
    nc["cap"] = cap_of(nc["n_cls"], params)
    key = ["d", "key"]
    m = nc.merge(default[key].assign(kept=True), on=key, how="left")
    kept = m["kept"].eq(True).to_numpy()
    dropped = m.loc[~kept, "cls"].value_counts()
    prot = m[kept & (m["rank"].to_numpy() > m["cap"].to_numpy())]
    return {"dropped": int((~kept).sum()), "dropped_by_class": dropped.to_dict(),
            "protected": len(prot), "protected_by_class": prot["cls"].value_counts().to_dict(),
            "pairs_protected": int(prot["d"].nunique())}


def overlap(default: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Default alternatives (d, key) found among the variant's (d, key):
    n_default, n_same, share."""
    a = set(zip(default["d"].astype(int), default["key"].astype(str)))
    b = set(zip(other["d"].astype(int), other["key"].astype(str)))
    same = len(a & b)
    return {"n_default": len(a), "n_same": same,
            "share": same / len(a) if a else math.nan}


# --- R3 on a subset ----------------------------------------------------------
def r3_keep(cands: pd.DataFrame, mu: float) -> np.ndarray:
    """The search's R3 on the candidates of ONE pair: keep iff gtt <= (1 +
    mu) x the least gtt of its mode class OR gtc <= (1 + mu) x the least gtc
    of its class, minima over ``cands`` (float32 values in float64, as
    rules.apply_rules compares them)."""
    cls = np.array([rules.mode_class(s) for s in cands["seq"].astype(str)], dtype=object)
    code = pd.factorize(cls, sort=True)[0]
    gtt = cands["gtt"].to_numpy(np.float32).astype(np.float64)
    gtc = cands["gtc"].to_numpy(np.float32).astype(np.float64)
    cut, _, _ = rules.r3_cut(code, gtt, gtc, float(mu))
    return ~cut


def r3_exact(cands: pd.DataFrame, mu_search: float, mu_rule: float) -> bool:
    """Whether R3 at ``mu_rule`` on a set kept by the search at ``mu_search``
    certainly equals R3 at ``mu_rule`` on all R0 candidates when gtc differs
    from the search's (frequency variants; ``cands`` carry the variant's
    gtc). A candidate outside the set had gtt > (1 + mu_search) x G, G the
    class minimum of gtt (which the set holds); its gtc' >= gtt. So when
    (1 + mu_search) x G >= (1 + mu_rule) x H', H' the least gtc' of the
    class in the set, no outside candidate holds the class minimum of gtc'
    or passes R3. True when this holds in every class."""
    cls = np.array([rules.mode_class(s) for s in cands["seq"].astype(str)], dtype=object)
    gtt = cands["gtt"].to_numpy(np.float32).astype(np.float64)
    gtc = cands["gtc"].to_numpy(np.float32).astype(np.float64)
    df = pd.DataFrame({"cls": cls, "gtt": gtt, "gtc": gtc}).groupby("cls")
    G, H = df["gtt"].min(), df["gtc"].min()
    return bool(((1 + mu_search) * G >= (1 + mu_rule) * H).all())


# --- Graph reach of the unserved pairs ---------------------------------------
def reach_levels(o: int, d: int, connectors: pd.DataFrame, links: pd.DataFrame,
                 t_ptr: np.ndarray, t_out: np.ndarray, max_links: int = 3) -> list:
    """Chains of 1 .. max_links service links from the access hubs of o in
    the graph of existing transfers (``t_ptr``, ``t_out``: CSR of
    ps-transfers rows that exist, by link_in), without timetables, R0 clauses
    or the 24-hour limit: per number of links, the distinct last links that
    end at an egress hub of d."""
    acc = connectors.loc[(connectors["fua"] == o)
                         & np.isfinite(connectors["acc_min"].to_numpy(np.float64)), "hub"]
    egr = connectors.loc[(connectors["fua"] == d)
                         & np.isfinite(connectors["egr_min"].to_numpy(np.float64)), "hub"]
    src, trg = links["src"].to_numpy(), links["trg"].to_numpy()
    level = np.flatnonzero(np.isin(src, acc.to_numpy()))
    out = []
    for k in range(1, max_links + 1):
        out.append(int(np.isin(trg[level], egr.to_numpy()).sum()))
        if k == max_links or not len(level):
            out += [0] * (max_links - k)
            break
        parts = [t_out[t_ptr[l]:t_ptr[l + 1]] for l in level]
        level = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
    return out


def transfer_csr(transfers: pd.DataFrame, n_links: int, min_active: int):
    """(ptr, link_out) of the ps-transfers rows that exist (basis 'D' with
    n_days >= MIN_ACTIVE_DATES, basis 'T' with n_days >= 1), by link_in."""
    b = transfers["basis"].astype(str).to_numpy()
    nd = transfers["n_days"].to_numpy(np.int64)
    ok = ((b == "D") & (nd >= min_active)) | ((b == "T") & (nd >= 1))
    li = transfers["link_in"].to_numpy(np.int64)[ok]
    lo = transfers["link_out"].to_numpy(np.int64)[ok]
    order = np.argsort(li, kind="stable")
    ptr = np.zeros(n_links + 1, dtype=np.int64)
    np.cumsum(np.bincount(li, minlength=n_links), out=ptr[1:])
    return ptr, lo[order]


def car_coverage(car: pd.DataFrame, universe) -> pd.DataFrame:
    """Per FUA of the universe: car rows out and in (ps-car)."""
    u = pd.Index(np.asarray(universe, dtype=np.int64), name="fua")
    return pd.DataFrame({"out": car.groupby("o").size().reindex(u, fill_value=0),
                         "in": car.groupby("d").size().reindex(u, fill_value=0)}).reset_index()


# --- Checks of the open items ------------------------------------------------
def passes_through(paths: pd.DataFrame, city: np.ndarray, p_ptr: np.ndarray,
                   p_city: np.ndarray, h_cols: tuple = ("b1", "a1", "b2", "a2", "b3", "a3")
                   ) -> np.ndarray:
    """Per path: some link of it passes on board (``p_ptr``, ``p_city``:
    search.passed_cities) o, d or the city of a hub of the path, the R0
    clause of search.py. ``paths`` need o, d, l1..l3 and the hub columns
    ``h_cols`` (view.link_hubs; -1 unused); ``city`` maps hub -> city. A
    link's passed cities never hold the cities of its own two hubs."""
    n = len(paths)
    forb = np.full((n, 2 + len(h_cols)), np.iinfo(np.int64).min, dtype=np.int64)
    forb[:, 0] = paths["o"].to_numpy(np.int64)
    forb[:, 1] = paths["d"].to_numpy(np.int64)
    for j, c in enumerate(h_cols):
        h = paths[c].to_numpy(np.int64)
        forb[:, 2 + j] = np.where(h >= 0, city[np.where(h >= 0, h, 0)], np.iinfo(np.int64).min)
    flag = np.zeros(n, dtype=bool)
    for k in (1, 2, 3):
        l = paths[f"l{k}"].to_numpy(np.int64)
        rows = np.flatnonzero(l >= 0)
        cnt = p_ptr[l[rows] + 1] - p_ptr[l[rows]]
        rep = np.repeat(rows, cnt)
        if not len(rep):
            continue
        start = np.repeat(p_ptr[l[rows]], cnt)
        offs = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        c = p_city[start + offs]
        hit = (forb[rep] == c[:, None]).any(axis=1)
        flag[np.unique(rep[hit])] = True
    return flag


def overshoots(paths: pd.DataFrame, connectors: pd.DataFrame) -> np.ndarray:
    """Per path: a hub before its last one (a1, b2, a2, b3 of
    view.link_hubs; -1 unused) has a finite egress to the destination d in
    ps-connectors, i.e. the path reaches a hub that serves d and rides on.
    R0 keeps station hubs of d out of the middle of a path, so in practice
    this is an airport of d's catchment."""
    eg = connectors[np.isfinite(connectors["egr_min"].to_numpy(np.float64))]
    base = np.int64(1) << np.int64(20)
    keys = np.unique(eg["hub"].to_numpy(np.int64) * base + eg["fua"].to_numpy(np.int64))
    out = np.zeros(len(paths), dtype=bool)
    for c in ("a1", "b2", "a2", "b3"):
        out |= _serves_d(paths, c, keys, base)
    return out


def _serves_d(paths: pd.DataFrame, col: str, keys: np.ndarray, base) -> np.ndarray:
    """The hub in column ``col`` is used, is not the path's last hub and has
    a finite egress to the path's destination."""
    h = paths[col].to_numpy(np.int64)
    last = paths["h_last"].to_numpy(np.int64)
    d = paths["d"].to_numpy(np.int64)
    return (h >= 0) & (h != last) & np.isin(h * base + d, keys)


def overshoot_kinds(paths: pd.DataFrame, connectors: pd.DataFrame,
                    link_mode: np.ndarray) -> pd.Series:
    """How a path flagged by ``overshoots`` uses the hub that serves d:
    'land_ground' it alights there (a1 or a2) and the next link is rail or
    coach; 'land_air' it alights there and every such landing is followed
    by a flight; 'board' it only boards there (b2 or b3: reached by ground
    and a move, then a flight); '' not flagged. ``link_mode`` maps link id ->
    mode (ps-links)."""
    eg = connectors[np.isfinite(connectors["egr_min"].to_numpy(np.float64))]
    base = np.int64(1) << np.int64(20)
    keys = np.unique(eg["hub"].to_numpy(np.int64) * base + eg["fua"].to_numpy(np.int64))
    f = {c: _serves_d(paths, c, keys, base) for c in ("a1", "b2", "a2", "b3")}

    def next_ground(k):
        l = paths[f"l{k}"].to_numpy(np.int64)
        m = np.asarray(link_mode, dtype=object)[np.where(l >= 0, l, 0)]
        return (l >= 0) & np.isin(m, ["B", "R"])
    ground = (f["a1"] & next_ground(2)) | (f["a2"] & next_ground(3))
    land = f["a1"] | f["a2"]
    board = ~land & (f["b2"] | f["b3"])
    out = np.full(len(paths), "", dtype=object)
    out[land & ~ground] = "land_air"
    out[ground] = "land_ground"
    out[board] = "board"
    return pd.Series(out, index=paths.index)


_R2_DETAIL = re.compile(r"^n (\d+) <= (\d+); \w+ ([-\d.e+]+) <= ([-\d.e+]+) \+ ([-\d.e+]+)")


def r2_details(detail: pd.Series) -> pd.DataFrame:
    """The numbers of R2 tests (rules._explain: 'n p <= q; T Tp <= Tq + eps;
    ...'): n_p, n_q, X_p, X_q, eps (NaN where the text does not match)."""
    m = detail.astype(str).str.extract(_R2_DETAIL)
    m.columns = ["n_p", "n_q", "X_p", "X_q", "eps"]
    return m.astype(float)


def oded_removals(details: pd.DataFrame, eps_abs: float) -> pd.Series:
    """Oded's case removed: the remover has fewer transfers and is later by
    more than the flat band EPS_T_ABS, so only the relative term of eps_T let
    it remove."""
    return (details["n_p"] < details["n_q"]) & (details["X_p"] - details["X_q"] > eps_abs)


def ftrans_callers(root: Path, files: list) -> list:
    """(file, line number, text) of every call of ftrans_table in ``files``."""
    out = []
    for name in files:
        p = Path(root) / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "ftrans_table(" in line and not line.lstrip().startswith("def "):
                out.append((name, i, line.strip()))
    return out


def remover_status(trace: pd.DataFrame) -> pd.Series:
    """Final status (kept, capped ...) of the remover named in 'by' of every
    R2 removal of ps-trace, looked up as 'cand' of the same pair."""
    r2 = trace[trace["rule"].astype(str) == "R2"]
    st = trace.set_index(["o", "d", "cand"])["status"].astype(str)
    got = st.reindex(list(zip(r2["o"], r2["d"], r2["by"]))).to_numpy(dtype=object)
    return pd.Series([str(s).split(":")[0] for s in got], index=r2.index)


def near_ties(alts: pd.DataFrame, gap: float) -> dict:
    """Kept service alternatives whose T lies at most ``gap`` minutes above
    the next faster kept alternative of the same pair and mode sequence: how
    many, in how many pairs, how many with the same first and last hub as
    that one, how many within 5 minutes."""
    s = alts[alts["seq"].astype(str) != "C"].sort_values(["o", "d", "seq", "T"], kind="stable")
    g = s.groupby(["o", "d", "seq"], sort=False)
    diff = (s["T"] - g["T"].shift(1)).to_numpy(np.float64)
    near = diff <= gap
    same = ((s["h_first"] == g["h_first"].shift(1))
            & (s["h_last"] == g["h_last"].shift(1))).to_numpy()
    return {"n": int(near.sum()), "pairs": int(s.loc[near, ["o", "d"]].drop_duplicates().shape[0]),
            "same_ends": int((near & same).sum()), "within5": int((diff <= 5).sum())}


# --- Distances between FUAs ----------------------------------------------------
def fua_centres(fuas: pd.DataFrame) -> pd.DataFrame:
    """lon and lat of every FUA from the WKB point in fuas.centre, indexed by
    id (NaN where the value is not a little-endian 2-D point)."""
    def xy(b):
        try:
            b = bytes(b)
            if len(b) >= 21 and b[0] == 1 and struct.unpack("<I", b[1:5])[0] == 1:
                return struct.unpack("<dd", b[5:21])
        except (TypeError, ValueError, struct.error):
            pass
        return (math.nan, math.nan)
    v = np.array([xy(b) for b in fuas["centre"]], dtype=np.float64).reshape(-1, 2)
    return pd.DataFrame({"lon": v[:, 0], "lat": v[:, 1]},
                        index=pd.Index(fuas["id"].to_numpy(np.int64), name="id"))


def gc_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km (haversine, earth radius 6,371 km)."""
    p1, p2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def pair_km(o, d, centres: pd.DataFrame) -> np.ndarray:
    """Distance between the centres of FUAs o and d (arrays of ids)."""
    c = centres
    return gc_km(c["lon"].reindex(o).to_numpy(), c["lat"].reindex(o).to_numpy(),
                 c["lon"].reindex(d).to_numpy(), c["lat"].reindex(d).to_numpy())


def band_labels(bins) -> list:
    """'0-150', ..., '1,500 and more' for the edges ``bins`` (km)."""
    out = [f"{a:,}-{b:,}" for a, b in zip(bins[:-2], bins[1:-1])]
    return out + [f"{bins[-2]:,} and more"]


def distance_table(funnel: pd.DataFrame, alts: pd.DataFrame, km, bins) -> pd.DataFrame:
    """Set sizes by distance between FUA centres (``km`` per funnel row):
    pairs, median kept, share of pairs where the cap acted, mean number of
    classes kept; kept service alternatives, the share of them whose T is at
    least twice the least T of the pair's kept service alternatives (as
    item 4), and the share that use a flight."""
    f = funnel[["o", "d", "n_kept", "n_capped", "n_classes"]].copy()
    f["band"] = pd.cut(np.asarray(km, float), bins, right=False, labels=band_labels(bins))
    s = alts.loc[alts["seq"].astype(str) != "C", ["o", "d", "seq", "T"]].copy()
    s["slow"] = (s["T"] >= 2 * s.groupby(["o", "d"])["T"].transform("min")).to_numpy()
    s["air"] = s["seq"].astype(str).str.contains("A").to_numpy()
    s = s.merge(f[["o", "d", "band"]], on=["o", "d"], how="left")
    g, h = f.groupby("band", observed=True), s.groupby("band", observed=True)
    out = pd.DataFrame({
        "pairs": g.size(), "median kept": g["n_kept"].median(),
        "capped": g["n_capped"].apply(lambda x: pct((x > 0).sum(), len(x), 1)),
        "classes (mean)": g["n_classes"].mean(), "service alternatives": h.size(),
        "2 x least T or more": h["slow"].apply(lambda x: pct(x.sum(), len(x), 1)),
        "with a flight": h["air"].apply(lambda x: pct(x.sum(), len(x), 1))})
    return out.rename_axis("km").reset_index()


def ratio_text(r: float) -> str:
    """A ratio with two decimals below 1, one below 100, none above."""
    return f"{r:.2f}" if r < 1 else (f"{r:.1f}" if r < 100 else f"{r:.0f}")


# --- The deck's counts, memory, variants --------------------------------------
def one_transfer_ratios(st: pd.DataFrame, deck: pd.DataFrame = DECK_TABLE1) -> pd.DataFrame:
    """Our R0 count against the deck's for its one-transfer sequences (``st``:
    seq_table indexed by seq): seq, ours, deck, ours / deck."""
    d = deck[deck["category"] != "Direct"].set_index("seq")["paths"]
    ours = st["n_r0"].reindex(d.index).to_numpy(np.float64)
    return pd.DataFrame({"seq": d.index.to_numpy(), "ours": ours, "deck": d.to_numpy(),
                         "ratio": ours / d.to_numpy(np.float64)})


def worker_estimate(peaks, workers: int) -> dict:
    """Memory of ``workers`` workers from the per-origin peaks (MB). peak_mb
    only grows within a worker, so a worker's peak is that of its last
    origin. Worst case: every worker at the largest peak. Estimate: one
    worker at the largest peak (it runs the largest origin) and the others
    at the largest peak below it."""
    p = pd.Series(peaks, dtype="float64").dropna()
    top = float(p.max())
    below = p[p < top]
    second = float(below.max()) if len(below) else top
    return {"top": top, "second": second, "n_top": int((p == top).sum()),
            "estimate": top + (workers - 1) * second, "worst": workers * top}


def rank_variants(sens: pd.DataFrame, origin: int, skip=("default", "no_cap")) -> list:
    """(variant, share_same_key) of one origin, the least share (the largest
    change of the set) first; ``skip`` left out."""
    s = sens[(sens["origin"] == origin) & ~sens["variant"].isin(skip)]
    s = s.sort_values(["share_same_key", "variant"], kind="stable")
    return list(zip(s["variant"].astype(str), s["share_same_key"].astype(float)))


def sens_code_hash(root: Path = io.ROOT) -> str:
    """First 16 hex digits of sha256 over the code that makes the
    sensitivity tables: the files the search runs (RUN_CODE), SENS_FILES,
    the source of the functions SENS_FUNCS of this module and the values of
    SENS_VARIANTS, SENS_ORIGINS and SENS_SEARCH_MU. Report text does not
    enter it."""
    h = hashlib.sha256()
    for name in list(RUN_CODE) + list(SENS_FILES):
        p = Path(root) / name
        h.update(name.encode("utf-8"))
        h.update(p.read_bytes() if p.exists() else b"missing")
    g = globals()
    for f in SENS_FUNCS:
        h.update(inspect.getsource(g[f]).encode("utf-8"))
    h.update(repr((SENS_VARIANTS, SENS_ORIGINS, SENS_SEARCH_MU)).encode("utf-8"))
    return h.hexdigest()[:16]


# --- Links and services --------------------------------------------------------
def suspect_links(links: pd.DataFrame, names, table: dict = SUSPECT_LINKS) -> tuple:
    """({link: what is wrong}, [stale ids]): the entries of ``table`` that
    ps-links still holds with the same board and alight hub (by name) and a
    riding time within 0.5 min, and those it does not."""
    lk = links.set_index("link")
    ok, stale = {}, []
    for l, (a, b, ride, why) in table.items():
        if (l in lk.index and str(names[int(lk.at[l, "src"])]) == a
                and str(names[int(lk.at[l, "trg"])]) == b
                and abs(float(lk.at[l, "ride"]) - ride) <= 0.5):
            ok[int(l)] = why
        else:
            stale.append(int(l))
    return ok, stale


def uses_links(paths: pd.DataFrame, ids) -> np.ndarray:
    """Per path: one of its links is in ``ids``."""
    ids = np.asarray(sorted(int(i) for i in ids), dtype=np.int64)
    out = np.zeros(len(paths), dtype=bool)
    for k in (1, 2, 3):
        out |= np.isin(paths[f"l{k}"].to_numpy(np.int64), ids)
    return out


def direct_summary(links: pd.DataFrame, sparse: pd.DataFrame, hubs: pd.DataFrame,
                   o: int, d: int, mode: str) -> dict:
    """Direct links of one mode from the hubs of FUA o to the hubs of FUA d
    (ps-links, and ps-sparse for those active on fewer than MIN_ACTIVE_DATES
    dates): count, departures a day (freq summed over the links), least and
    largest freq, active dates and the date range."""
    ho, hd = hubs.loc[hubs["fua"] == o, "hub"], hubs.loc[hubs["fua"] == d, "hub"]

    def pick(t):
        return t[t["src"].isin(ho) & t["trg"].isin(hd) & (t["mode"].astype(str) == mode)]
    l, s = pick(links), pick(sparse)
    out = {"links": len(l), "per_day": float(l["freq"].sum()), "sparse": len(s),
           "sparse_dates": int(s["active_dates"].max()) if len(s) else 0}
    if len(l):
        out.update(f_min=float(l["freq"].min()), f_max=float(l["freq"].max()),
                   dates_min=int(l["active_dates"].min()), dates_max=int(l["active_dates"].max()),
                   first=D.index_date(int(l["first_day"].min())),
                   last=D.index_date(int(l["last_day"].max())))
    return out


def summary_text(s: dict, word: str) -> str:
    """direct_summary in words."""
    if not s["links"]:
        return f"no {word} link" + (f" (ps-sparse {s['sparse']}, at most {s['sparse_dates']} "
                                    "dates)" if s["sparse"] else "")
    each = (f"{s['f_min']:g} a day" if s["f_min"] == s["f_max"] else
            f"{s['f_min']:g} to {s['f_max']:g} a day")
    dates = (f"{s['dates_min']}" if s["dates_min"] == s["dates_max"] else
             f"{s['dates_min']} to {s['dates_max']}")
    many = s["links"] > 1
    return (f"{s['links']} {word} link{'s' if many else ''}, {each}"
            + (f" each ({s['per_day']:g} in all)" if many else "")
            + f", on {dates} dates between {s['first']} and {s['last']}"
            + (f"; ps-sparse {s['sparse']} more" if s["sparse"] else ""))


def dwell_table(timetable: pd.DataFrame, journeys: pd.DataFrame, stations: pd.DataFrame,
                feeds: pd.DataFrame, a_pat: str, b_pat: str) -> pd.DataFrame:
    """Journeys of data-upstream-hubs that call at a station whose name
    matches ``a_pat`` and later at one that matches ``b_pat``, by feed and
    dwell at the first of the two (dep - arr, minutes): journeys, least
    riding minutes from the departure and from the arrival there."""
    sa = stations.loc[stations["name"].astype(str).str.contains(a_pat, regex=True), "stn"]
    sb = stations.loc[stations["name"].astype(str).str.contains(b_pat, regex=True), "stn"]
    a = timetable[timetable["stn"].isin(sa)].merge(journeys[["jrn", "feed"]], on="jrn")
    b = timetable.loc[timetable["stn"].isin(sb), ["jrn", "arr"]].rename(columns={"arr": "a2"})
    m = a.merge(b, on="jrn")
    m = m[m["a2"] > m["dep"]].copy()
    m["dwell"] = m["dep"] - m["arr"]
    m["ride"] = m["a2"] - m["dep"]
    m["ride_arr"] = m["a2"] - m["arr"]
    g = (m.groupby(["feed", "dwell"]).agg(journeys=("jrn", "size"), ride=("ride", "min"),
                                          ride_from_arr=("ride_arr", "min")).reset_index())
    g = g.merge(feeds.rename(columns={"id": "feed", "name": "feed_name"}), on="feed", how="left")
    return g.sort_values(["ride", "feed"], kind="stable").reset_index(drop=True)


def direct_fate(m: str, alts: pd.DataFrame, best: pd.DataFrame, params: dict) -> str:
    """Why the direct mode ``m`` of a pair has R0 candidates but no kept
    alternative (alts: kept rows of the pair; best: its ps-best-by-sequence
    rows). R2 within a class needs fewer transfers, or a T below the other's
    by more than eps_T (rules.dominance); so no alternative removes the
    fastest direct alternative of a mode, and when any direct candidate of
    the mode passes R3 one survives R2 and only the cap can drop it. R3 is
    certain to keep the candidate with the least gtt of the sequence when
    its gtt is within (1 + MU) x the least gtt of the mode class (the class
    minimum is over ps-best-by-sequence); otherwise the stored tables cannot
    tell whether a more frequent direct candidate passed R3 on gtc."""
    if params.get("CROSS_CLASS"):
        return "the stored tables do not show why (CROSS_CLASS on)"
    cls = np.array([rules.mode_class(s) if s != "C" else "C" for s in best["seq"].astype(str)])
    g_cls = float(best.loc[cls == m, "gtt"].astype(np.float32).min())
    b = best[best["seq"].astype(str) == m].iloc[0]
    margin = (1 + float(params["MU"])) * g_cls
    if float(np.float32(b["gtt"])) > margin:
        return (f"its least gtt ({float(b['gtt']):.0f}) is above {1 + float(params['MU']):g} x "
                f"the least gtt of class {m} ({g_cls:.0f}), so R3 cut it unless a more "
                "frequent direct candidate passed on gtc (not stored)")
    a = alts[alts["seq"].astype(str) != "C"]
    alone = a.groupby("cls")["cls"].transform("size") == 1
    above = a[alone & (a["gtc"] > float(b["gtc"]))].sort_values("gtc")
    w = [f"{s} (gtc {g:.0f})" for s, g in zip(above["seq"], above["gtc"])]
    keep = w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1] if w else ""
    return (f"it passed R3 (least gtt {float(b['gtt']):.0f}, within "
            f"{1 + float(params['MU']):g} x {g_cls:.0f}) and R2 cannot remove the fastest "
            "direct path of a mode, so the cap dropped it"
            + (f", while it kept {keep} above it, each the last of its class (R4; item 4)"
               if len(above) else ""))


# --- Sensitivity (slow: one search per origin) -------------------------------
def sensitivity_specs() -> dict:
    """Name -> spec of every sensitivity variant (view.variant_spec form)."""
    return dict(SENS_VARIANTS)


def _nanmax(*values) -> float:
    """Largest of the values that are not NaN; NaN when all are."""
    v = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return max(v) if v else math.nan


def sensitivity_pair(cands: pd.DataFrame, params: dict, specs: dict) -> dict:
    """Every variant of ``specs`` (the first is "default") on the candidates
    of ONE pair, numbered as search.decide numbers them (view.number) and
    carrying the link frequencies of view.link_freqs.

    Returns {"variants": {name: {"alts": the kept alternatives
    (rules.ALTERNATIVES), "keys": their link keys, "n_capped", "same_rep"
    and "found" (default representatives that are representatives, or
    members, under the variant), "exact" (frequency variants: r3_exact on
    the variant's gtc; None for the others), "seconds" (the rules)}},
    "kept_r3": candidates R3 keeps under the default, "r2": the default's
    R2 removals {"removals", "oded" (oded_removals), "oded_eps_max" and
    "eps_max" (the largest band among those and among all removals),
    "kept_eps_max" (the largest band of a class with a kept service
    alternative)}}.
    """
    if next(iter(specs)) != "default":
        raise ValueError("the first variant must be 'default' (the others are compared "
                         "with it)")
    key_of = cands.set_index("cand")["key"]
    out, base, kept_r3 = {}, None, None
    r2 = {"removals": 0, "oded": 0, "oded_eps_max": math.nan, "eps_max": math.nan,
          "kept_eps_max": math.nan}
    for name, spec in specs.items():
        t = time.perf_counter()
        alts, trace = V.decide_variant(cands, params, spec)
        secs = time.perf_counter() - t
        status = trace["status"].astype(str)
        if name == "default":
            base = alts
            kept_r3 = int((status != "cut_r3").sum())
            rem = trace[trace["rule"].astype(str) == "R2"]
            if len(rem):
                det = r2_details(rem["detail"])
                od = oded_removals(det, float(params["EPS_T_ABS"])).to_numpy()
                r2.update(removals=len(rem), oded=int(od.sum()),
                          oded_eps_max=_nanmax(*det.loc[od, "eps"]),
                          eps_max=_nanmax(*det["eps"]))
            svc = alts[alts["cls"].astype(str) != "C"]      # the car's band is moot
            if len(svc):
                r2["kept_eps_max"] = _nanmax(*svc["eps_T"].to_numpy(float))
        exact = None
        if name in FREQ_VARIANTS:
            p = {**params, **spec.get("params", {})}
            seen = V.with_frequency(cands, p, spec.get("freq", "path"))
            exact = r3_exact(seen, SENS_SEARCH_MU, float(p["MU"]))
        reps = set(base["alt"].astype(int))
        out[name] = {"alts": alts, "keys": [str(key_of[a]) for a in alts["alt"].astype(int)],
                     "n_capped": int((status == "capped").sum()),
                     "same_rep": len(reps & set(alts["alt"].astype(int))),
                     "found": len(reps & V.members(alts)), "exact": exact, "seconds": secs}
    return {"variants": out, "kept_r3": kept_r3, "r2": r2}


def run_sensitivity(net, o: int, params: dict, links: pd.DataFrame,
                    stored: pd.DataFrame, funnel_o: pd.DataFrame,
                    specs: dict | None = None, log=io.log,
                    max_r3: int = SENS_MAX_R3) -> dict:
    """Every variant of ``specs`` on the candidates of origin ``o`` (module
    docstring; ``sensitivity_pair`` per destination). ``stored``:
    ps-alternatives rows of o; ``funnel_o``: its ps-funnel rows (n_r3 is
    checked against R3 of the default variant). Refuses an origin whose
    ps-funnel n_r3 (MU = 0.5) exceeds ``max_r3``: the search at MU = 1.0
    builds the rows of all destinations at once.

    Returns {"rows": one dict per variant, "alts": DataFrame of every
    alternative (origin, variant, d, rank, seq, cls, key, T, gtt, gtc, freq,
    n_variants), "run": dict of the search, the checks, the R2 removals of
    the default and the memory of this process after the origin,
    "mismatch": DataFrame of default alternatives that differ from
    ``stored``}.
    """
    specs = sensitivity_specs() if specs is None else specs
    if next(iter(specs)) != "default":
        raise ValueError("the first variant must be 'default' (the others are compared "
                         "with it)")
    n3_total = int(funnel_o["n_r3"].sum())
    if n3_total > max_r3:
        raise ValueError(f"origin {o}: ps-funnel n_r3 {n3_total:,} exceeds {max_r3:,}; the "
                         "search at MU = 1.0 would hold about three times as many rows")
    t0 = time.perf_counter()
    dests = [int(x) for x in net.fuas if int(x) != int(o)]
    c = search.candidates(net, int(o), dests, params, SENS_SEARCH_MU)
    t_search = time.perf_counter() - t0
    c = V.link_freqs(c, links)
    log(f"sensitivity {o}: {len(c):,} candidates at MU {SENS_SEARCH_MU} in {t_search:.1f} s")
    n3 = funnel_o.set_index("d")["n_r3"]
    secs = {k: 0.0 for k in specs}
    alts_out, per_pair = [], []
    r3_bad, n_pairs = [], 0
    agg = {"removals": 0, "oded": 0, "oded_eps_max": math.nan, "eps_max": math.nan,
           "kept_eps_max": math.nan}
    exact_fail = {k: 0 for k in specs if k in FREQ_VARIANTS}
    for d, grp in c.groupby("d", sort=True):
        res = sensitivity_pair(V.number(grp), params, specs)
        n_pairs += 1
        if res["kept_r3"] != int(n3.get(int(d), -1)):
            r3_bad.append((int(d), res["kept_r3"], int(n3.get(int(d), -1))))
        for k in ("removals", "oded"):
            agg[k] += res["r2"][k]
        for k in ("oded_eps_max", "eps_max", "kept_eps_max"):
            agg[k] = _nanmax(agg[k], res["r2"][k])
        for name, v in res["variants"].items():
            alts = v["alts"]
            secs[name] += v["seconds"]
            if v["exact"] is False:
                exact_fail[name] += 1
            per_pair.append((name, int(d), len(alts), v["n_capped"], v["same_rep"], v["found"]))
            alts_out.append(pd.DataFrame({
                "origin": np.int16(o), "variant": name, "d": np.int16(d),
                "rank": np.arange(1, len(alts) + 1, dtype=np.int16),
                "seq": alts["seq"].astype(str).to_numpy(),
                "cls": alts["cls"].astype(str).to_numpy(),
                "key": v["keys"],
                "T": alts["T"].to_numpy(np.float32), "gtt": alts["gtt"].to_numpy(np.float32),
                "gtc": alts["gtc"].to_numpy(np.float32),
                "freq": alts["freq"].to_numpy(np.float32),
                "n_variants": alts["n_variants"].to_numpy(np.int32)}))
    alts_df = pd.concat(alts_out, ignore_index=True)
    pp = pd.DataFrame(per_pair, columns=["variant", "d", "n_alt", "n_capped", "same_rep",
                                         "found"])
    mismatch = default_mismatch(alts_df[alts_df["variant"] == "default"], stored)
    default = alts_df[alts_df["variant"] == "default"]
    rows = []
    for name in specs:
        x = pp[pp["variant"] == name]
        a = alts_df[alts_df["variant"] == name]
        ov = overlap(default, a)
        rows.append({
            "origin": int(o), "variant": name,
            "changes": ";".join(f"{k}={v}" for k, v in specs[name].get("params", {}).items())
            + (f"freq={specs[name]['freq']}" if "freq" in specs[name] else ""),
            "pairs": len(x), "n_alt": int(x["n_alt"].sum()),
            "per_pair": float(x["n_alt"].mean()) if len(x) else math.nan,
            "pairs_capped": int((x["n_capped"] > 0).sum()), "n_capped": int(x["n_capped"].sum()),
            "before_cap": int(x["n_alt"].sum() + x["n_capped"].sum()),
            "n_same_key": ov["n_same"], "share_same_key": ov["share"],
            "n_found_member": int(x["found"].sum()),
            "share_found_member": (float(x["found"].sum()) / len(default)
                                   if len(default) else math.nan),
            **{f"cls_{k}": int((a["cls"] == k).sum()) for k in CLASSES},
            "pairs_not_exact": exact_fail.get(name, 0),
            "seconds": round(secs[name], 2)})
    run = {"origin": int(o), "search_mu": SENS_SEARCH_MU, "search_s": round(t_search, 1),
           "candidates": len(c), "pairs": n_pairs, "n_r3_mu05": n3_total,
           "r3_check": "pass" if not r3_bad else f"FAIL on {len(r3_bad)} pairs",
           "r3_bad": json.dumps(r3_bad[:20]),
           "default_check": "pass" if not len(mismatch) else f"FAIL: {len(mismatch)} rows",
           "r2_removals": agg["removals"], "oded_removals": agg["oded"],
           "oded_eps_max": agg["oded_eps_max"], "r2_eps_max": agg["eps_max"],
           "eps_max": agg["kept_eps_max"],
           "stored_alternatives": len(stored), "default_alternatives": len(default),
           "rss_mb": search.rss_mb(), "peak_mb": search.peak_mb()}
    return {"rows": rows, "alts": alts_df, "run": run, "mismatch": mismatch.assign(origin=o)}


def default_mismatch(default: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    """Rows of the default variant and of the stored ps-alternatives of one
    origin that differ on (d, rank, key, seq, T, gtt); empty when equal. T and
    gtt are compared as float32, as stored."""
    cols = ["d", "rank", "key", "seq", "T", "gtt"]
    a = default[cols].copy()
    b = stored[cols].copy()
    for x in (a, b):
        x["d"] = x["d"].astype(np.int64)
        x["rank"] = x["rank"].astype(np.int64)
        x["key"] = x["key"].astype(str)
        x["seq"] = x["seq"].astype(str)
        x["T"] = x["T"].astype(np.float32)
        x["gtt"] = x["gtt"].astype(np.float32)
    m = a.merge(b, on=cols, how="outer", indicator=True)
    return m[m["_merge"] != "both"].rename(columns={"_merge": "side"}).reset_index(drop=True)


# --- Loading -----------------------------------------------------------------
def load_inputs(log_path: Path | None = None) -> dict:
    """Every stored table the report reads (selected columns), the manifest
    records, the shard attrs and the console log when present."""
    t0 = time.perf_counter()
    x = {"records": V.manifest_records()}
    x["run"], x["combine"] = run_records(x["records"])
    x["params"] = dict(x["run"]["params"]) if x["run"] else dict(io.P)
    x["attrs"] = load_shard_attrs()
    x["fuas"] = io.load("fuas", io.DATA, ["id", "name", "icc", "centre"])
    x["hubs"] = io.load_out("ps-hubs", ["hub", "kind", "name", "fua", "iata", "icc", "tz_std"])
    x["names"] = V.hub_names(x["hubs"])
    x["links"] = io.load_out("ps-links", ["link", "src", "trg", "mode", "cls", "ride", "km",
                                          "ev0", "ev1", "active_dates", "first_day",
                                          "last_day", "freq", "freq_max", "freq_span",
                                          "freq_fixed", "active_share"])
    x["funnel"] = io.load_out("ps-funnel")
    x["seq_counts"] = io.load_out("ps-seq-counts")
    x["alts"] = io.load_out("ps-alternatives")
    x["best"] = io.load_out("ps-best-by-sequence", ["o", "d", "seq", "l1", "l2", "l3", "acc",
                                                     "egr", "T", "gtt", "gtc"])
    x["unserved"] = io.load_out("ps-unserved")
    x["car"] = io.load_out("ps-car")
    x["connectors"] = io.load_out("ps-connectors")
    x["moves"] = io.load_out("ps-moves")
    x["members"] = io.load_out("ps-link-members", ["link", "agency", "route_type"])
    x["trace"] = io.load_out("ps-trace", ["o", "d", "cand", "seq", "n", "T", "cls", "status",
                                          "rule", "by", "detail"])
    x["sparse"] = io.load_out("ps-sparse", ["src", "trg", "mode", "ride", "active_dates",
                                            "first_day", "last_day"])
    x["city"] = search.hub_cities(x["hubs"], x["connectors"])
    if log_path is not None and Path(log_path).exists():
        start = (dt.datetime.strptime(x["run"]["time"], "%Y-%m-%d %H:%M:%S")
                 - dt.timedelta(seconds=float(x["run"]["info"]["seconds"]))).date()
        x["log"] = parse_log(Path(log_path).read_text(encoding="utf-8"), start)
        x["log_path"] = Path(log_path)
    else:
        x["log"] = None
    for name, key in ((SENS_TABLE, "sens"), (SENS_ALTS, "sens_alts"), (SENS_RUNS, "sens_runs"),
                      (SENS_MISMATCH, "sens_mismatch")):
        p = io.OUT / f"{name}.parquet"
        x[key] = pd.read_parquet(p) if p.exists() else None
    stage_b = io.OUT / "stage-b" / "variants.parquet"
    x["stage_b"] = pd.read_parquet(stage_b) if stage_b.exists() else None
    x["review"] = load_review_inputs()
    x["load_s"] = time.perf_counter() - t0
    return x


# Station pairs of the dwell tables (item "eurostar" and "fast_links"): key,
# name pattern of the first station, of the later one (data-upstream-hubs)
DWELL_PAIRS = [("eurostar", "Paris Gare du Nord|Paris-Nord|Paris Nord", "Pancras"),
               ("eurostar_back", "Pancras", "Paris Gare du Nord|Paris-Nord|Paris Nord"),
               ("lyon", "Lyon Part", "Paris Gare de Lyon|Paris Lyon")]


def load_review_inputs() -> dict:
    """The network inputs that only section 11 reads: the dwell tables of
    DWELL_PAIRS (data-upstream-hubs ic-stations, ic-timetable, ic-journeys,
    feed-info; only the rows of those stations are read) and the carriers
    of data-upstream air-timetable. A file that cannot be read leaves None
    and a note."""
    out = {"dwell": {}, "carriers": None, "notes": []}
    try:
        stations = pd.read_parquet(io.HUBS / "ic-stations.parquet", columns=["stn", "name"])
        feeds = pd.read_parquet(io.HUBS / "feed-info.parquet", columns=["id", "name"])
        names = stations["name"].astype(str)
        hit = np.zeros(len(stations), dtype=bool)
        for _, a, b in DWELL_PAIRS:
            hit |= names.str.contains(a, regex=True).to_numpy()
            hit |= names.str.contains(b, regex=True).to_numpy()
        ids = stations.loc[hit, "stn"].astype(int).tolist()
        tt = pq.read_table(io.HUBS / "ic-timetable.parquet", columns=["jrn", "stn", "arr", "dep"],
                           filters=[("stn", "in", ids)]).to_pandas()
        jr = pq.read_table(io.HUBS / "ic-journeys.parquet", columns=["jrn", "feed"],
                           filters=[("jrn", "in", tt["jrn"].unique().tolist())]).to_pandas()
        for key, a, b in DWELL_PAIRS:
            out["dwell"][key] = dwell_table(tt, jr, stations, feeds, a, b)
    except (OSError, KeyError, ValueError) as e:
        out["notes"].append(f"data-upstream-hubs not read ({type(e).__name__})")
    try:
        at = pq.read_table(io.DATA / "air-timetable.parquet", columns=["carrier", "src", "trg"],
                           read_dictionary=["carrier", "src", "trg"]).to_pandas()
        counts = at["carrier"].value_counts()
        counts = counts[counts > 0]
        routes = {}
        for s, t in CARRIER_ROUTES:
            r = at.loc[(at["src"] == s) & (at["trg"] == t), "carrier"].value_counts()
            routes[(s, t)] = {str(k): int(v) for k, v in r[r > 0].items()}
        out["carriers"] = {"codes": sorted(str(k) for k in counts.index),
                           "absent": {k: int(counts.get(k, 0)) for k in CARRIERS_ABSENT},
                           "routes": routes}
        del at
    except (OSError, KeyError, ValueError) as e:
        out["notes"].append(f"data-upstream air-timetable not read ({type(e).__name__})")
    return out


def load_transfer_csr(n_links: int, min_active: int):
    """``transfer_csr`` of ps-transfers (basis read as a dictionary column)."""
    t = pq.read_table(io.OUT / "ps-transfers.parquet",
                      columns=["link_in", "link_out", "basis", "n_days"],
                      read_dictionary=["basis"]).to_pandas()
    return transfer_csr(t, n_links, min_active)


def load_passed(x: dict, share: float):
    """search.passed_cities of ps-links at ``share`` (events and their dates
    from ps-events and ps-event-masks)."""
    ev = io.load_out("ps-events", ["dep", "ride", "trip"])
    masks = io.load_npy("ps-event-masks")
    lk = x["links"].sort_values("link")
    out = search.passed_cities(lk["src"].to_numpy(), lk["trg"].to_numpy(),
                               lk["ev0"].to_numpy(), lk["ev1"].to_numpy(),
                               ev["dep"].to_numpy(), ev["ride"].to_numpy(),
                               ev["trip"].to_numpy(), x["city"], search.event_dates(masks),
                               share)
    del masks
    return out


def name_of(x: dict, fua: int) -> str:
    return V.fua_name(x["fuas"], int(fua))


def with_routes(paths: pd.DataFrame, x: dict) -> pd.DataFrame:
    """Paths with the hub columns and their route text (view.route)."""
    lk = x["links"].sort_values("link")
    p = V.link_hubs(paths, lk["src"].to_numpy(), lk["trg"].to_numpy())
    p["route"] = [V.route(r, x["names"]) for _, r in p.iterrows()]
    return p


def resolve_pairs(fuas: pd.DataFrame, pairs=WELL_KNOWN_PAIRS) -> list:
    """(o, d, name used for o, name used for d, note) per pair: the name, else
    its LOCAL_NAMES form; a pair that resolves neither way is reported."""
    out = []
    for a, b in pairs:
        ids, used = [], []
        for n in (a, b):
            for cand in (n, LOCAL_NAMES.get(n)):
                if cand is None:
                    continue
                try:
                    ids.append(V.resolve_fua(cand, fuas))
                    used.append(cand)
                    break
                except ValueError:
                    continue
            else:
                ids.append(None)
                used.append(n)
        out.append((ids[0], ids[1], used[0], used[1], a, b))
    return out


# --- Report sections ---------------------------------------------------------
SHOWN = ["MU", "EPS_T_ABS", "EPS_T_REL", "RHO_F", "R1_KEY", "R2_TIME", "R2_GATE",
         "CROSS_CLASS", "R4_GUARANTEE", "K_CAP", "K_EXTRA", "PASSED_SHARE", "WORKERS",
         "WINDOWS", "MAX_TRANSFERS", "MODE_WORDS", "T_MAX_MIN", "MIN_ACTIVE_DATES",
         "MIN_COMMON_DATES", "TYPICAL_SHARE", "ACCESS_MAX_AIR_MIN", "INCLUDE_CAR",
         "BETA_HWT", "HWT_MAX", "MEM_CAND_GB"]


def section_run(x: dict, ctx: dict) -> str:
    run, comb, p = x["run"], x["combine"], x["params"]
    info = run["info"]
    hashes = sorted({a.get("param_hash") for a in x["attrs"].values()})
    log = x["log"]
    start = log["start"] if log and log.get("start") else (
        dt.datetime.strptime(run["time"], "%Y-%m-%d %H:%M:%S")
        - dt.timedelta(seconds=float(info["seconds"])))
    ctx["run_start"] = start
    g = ctx.get("git") or git_info(paths=tuple(RUN_CODE))
    times = file_times(io.ROOT, RUN_CODE)
    latest = times.sort_values("modified").iloc[-1]
    recomputed = run_hash({**io.P}, bool(info["named"]), input_files())
    from_manifest = run_hash(dict(run["params"]), bool(info["named"]), input_files())
    changed = [k for k in SHOWN if str(io.P.get(k)) != str(p.get(k))]
    lines = ["## 1. What was run", ""]
    lines += [
        f"Stage D ran pathset-enumerate.py on {len(info['origins_run']):,} origins with "
        f"{info['workers']} worker processes, from {start:%Y-%m-%d %H:%M:%S} to "
        f"{run['time']} (ps-manifest.jsonl, the 'enumerate' record of {run['time']}: "
        f"origins_run {len(info['origins_run']):,}, origins_skipped "
        f"{len(info['origins_skipped'])}, named {info['named']}, MU {info['mu']}, "
        f"PASSED_SHARE {info.get('passed_share')}). The shards were combined at "
        f"{comb['time'] if comb else 'n/a'} (record with 'combined'). "
        f"Python {run['versions']['python']}, numpy {run['versions']['numpy']}, pandas "
        f"{run['versions']['pandas']}, numba {run['versions']['numba']} "
        f"({run['python']}).", ""]
    lines += [
        f"Run hash: {', '.join(hashes)} in the attrs of all {len(x['attrs']):,} funnel "
        f"shards (paths/funnel/o*.parquet, param_hash)"
        + (f"; the console log gives {log['hash']}" if log and log.get("hash") else "")
        + ". Recomputed now from pathset-params.yml and the present input files: "
        f"{recomputed}; from the parameters stored in the manifest record: "
        f"{from_manifest}. "
        + ("Both equal the shard hash, so neither the parameters nor the input files "
           "(size and modification time) have changed since the run."
           if recomputed == from_manifest == hashes[0] and len(hashes) == 1 else
           "They differ from the shard hash: the parameters or the inputs changed after "
           "the run."), ""]
    status = g["status"]
    lines += [
        f"Code: branch {g['branch']}, HEAD {g['head']} (committed {g['date']}). "
        f"git status --porcelain on the files the search runs (listed below): "
        + ("nothing to report." if status == "" else
           (f"'{status}'." if status else "git not available.")) + " "
        f"The run hash covers parameters and inputs, not code (open item 9). The files "
        f"the search runs ({', '.join(RUN_CODE)}) were last modified at "
        f"{latest['modified']:%Y-%m-%d %H:%M:%S} ({latest['file']}), "
        + ("before" if latest["modified"] < start else "AFTER")
        + f" the run started at {start:%H:%M:%S}. "
        + (f"So the shards come from the code that {g['head']} holds"
           + (f": the commit came {(dt.datetime.strptime(g['date'][:19], '%Y-%m-%d %H:%M:%S') - start).total_seconds() / 60:.0f} "
              "minutes after the start, with the files as they were on disk when the run "
              "began." if g["date"] and g["date"][:19] > start.strftime("%Y-%m-%d %H:%M:%S")
              else ".")
           if latest["modified"] < start and status == "" else
           "The code that ran cannot be tied to a commit."), ""]
    rows = [(k, str(p.get(k))) + ((f"now {io.P.get(k)}" if k in changed else ""),)
            for k in SHOWN]
    cols = ["parameter", "value", "differs now"]
    tab = pd.DataFrame(rows, columns=cols)
    if not changed:
        tab = tab.drop(columns="differs now")
    lines += ["Parameters that matter, as stored in the manifest record of the run "
              "(pathset-params.yml at the time; "
              + ("no value differs from the file now" if not changed else
                 "the last column flags values that differ in the file now") + "):", "",
              md(tab), "",
              "WINDOWS are the transfer windows in minutes from arrival to departure "
              "(A = air, P = bus or rail; deck slide 234). MODE_WORDS air_block_max2 = 36 "
              "mode sequences (one air block, at most two flights).", "",
              "The rules (pathset/rules.py). R3, the margin, comes first: a candidate is "
              "cut when both its gtt and its gtc exceed (1 + MU) x the least of its mode "
              "class. R1 groups near-ties of one mode sequence: T within eps_T = "
              f"max(EPS_T_ABS, EPS_T_REL x the least T of the class) of a leader. R2 "
              "removes, inside a mode class, an alternative q when another p has no more "
              "transfers, T_p <= T_q + eps_T, and either fewer transfers or T_p < T_q - "
              "eps_T, and p passes the frequency gate. R4 keeps the last alternative of a "
              "class. R5 caps the set at max(K_CAP, classes present + K_EXTRA), dropping "
              "the highest gtc first.", "",
              f"The gate of this run, R2_GATE {p.get('R2_GATE')}: p may remove q only if "
              f"freq(p) >= RHO_F x the largest path frequency among the members of q's R1 "
              f"group (RHO_F {p.get('RHO_F')}), so no member of q's group runs more than "
              "1/RHO_F times as often as its remover. The other forms: 'path' compares "
              "with the frequency of q's representative, 'pooled' (the plan's wording) "
              "with the pooled frequency of q's group (pathset-params.yml; rules.dominance)."
              if p.get("R2_GATE") == "top" else
              f"The gate of this run: R2_GATE {p.get('R2_GATE')} (pathset-params.yml; "
              "rules.dominance).", ""]
    inp = run["inputs"]
    now = io.fingerprint([Path(k) for k in inp])
    same = all(now.get(k) == v for k, v in inp.items())
    rows = [(Path(k).name, f"{v[0] / 2 ** 20:,.1f}",
             dt.datetime.fromtimestamp(v[1]).strftime("%Y-%m-%d %H:%M")) for k, v in inp.items()]
    lines += [f"Inputs (ps-manifest.jsonl, 'inputs' of the run record; size and time "
              f"{'unchanged' if same else 'CHANGED'} now):", "",
              md(pd.DataFrame(rows, columns=["file", "MB", "modified"])), ""]
    sizes = comb["info"]["combined"] if comb else {}
    tables = [
        ("ps-alternatives", "kept alternatives: one row per alternative, the PATH "
         "columns of its representative, rank (1.. in increasing gtc), cls, "
         "freq_pooled, n_variants, status"),
        ("ps-variants", f"other members of each alternative (at most VARIANTS_KEPT = "
         f"{p.get('VARIANTS_KEPT')} each), member 1.. in increasing (T, gtt, key)"),
        ("ps-best-by-sequence", "per (o, d, mode sequence) the R0 candidate with the "
         "least gtt, before R3 (input of the CASPT curves); the car is seq C"),
        ("ps-funnel", "per ordered FUA pair: n_r0, n_r0_bound, n_r3, n_r1, n_r2, "
         "n_kept, n_capped, n_classes, best_T, best_gtt, car_min, pt (section 4)"),
        ("ps-seq-counts", "per origin and mode sequence: n_r0, n_r0_bound, n_r3, n_kept"),
        ("ps-unserved", "pairs without a service alternative (section 7)"),
        ("ps-trace", "named pairs only: every R0 candidate (search at MU = inf) with its "
         "fate under the rules"),
    ]
    rows = [(t, n_(sizes.get(t, math.nan)), d) for t, d in tables]
    shards = {t: len(list((io.OUT / "paths" / t).glob("o*.parquet")))
              for t in ("alternatives", "variants", "best", "funnel", "seq_counts", "trace")}
    lines += [f"Plan, Verification 4, '384 manifest rows': the manifest record lists "
              f"{len(info['origins_run']):,} origins run and none skipped; shards per table "
              "in paths/: " + ", ".join(f"{k} {v}" for k, v in shards.items())
              + f"; ps-funnel has {n_(sizes.get('ps-funnel', math.nan))} rows "
              f"({len(info['origins_run'])} x {len(info['origins_run']) - 1} = "
              f"{len(info['origins_run']) * (len(info['origins_run']) - 1):,}).", ""]
    lines += ["Outputs (rows as written by the combine step):", "",
              md(pd.DataFrame(rows, columns=["table", "rows", "holds"])), "",
              "PATH columns: o, d, seq, n (transfers), h_first, h_last, l1-l3 (ps-links "
              "ids), cities (route pattern), key (link ids), acc and egr (car access and "
              "egress, clock minutes), mv1, mv2, ft1, ft2 (moves and transfer minutes), "
              "t_a .. t_e (the deck's weighted terms, sum gtt), T (clock minutes door to "
              "door), gtt, hwt, hwt_first, gtc, freq (path departures per day), "
              "freq_counted, basis (D dates, T typical days), n_days "
              "(pathset/schemas.py).", "",
              "To see one pair: `python -X utf8 pathset-view.py \"Amsterdam\" \"Berlin\"` "
              "(re-enumerates the pair at MU = inf and prints the funnel, the kept "
              "alternatives with the deck's components and the removed ones; options "
              "--r2-gate, --r1-key, --eps, --mu, --rho, --freq).", ""]
    return "\n".join(lines)


def section_stage_c(x: dict, ctx: dict) -> str:
    p = x["params"]
    origins = [int(o) for o in p["STAGE_C_ORIGINS"]]
    tt = ctx["timing"].set_index("origin")
    f = x["funnel"]
    named_o = {int(a) for a, _ in p["NAMED_PAIRS"]}
    rows, cost = [], []
    for o in origins:
        s = tt.loc[o]
        fo = f[f["o"] == o]
        lo, hi = r0_interval(fo["n_r0"].sum(), fo["n_r0_bound"].sum())
        rows.append({"origin": f"{name_of(x, o)} ({o})", "total s": s["total_s"],
                     "pass 1 s": s["pass1_s"], "pass 2 s": s["pass2_s"],
                     "trace s": s["trace_s"], "rules s": s["rules_s"],
                     "rows etc. s": s["other_s"], "blocks": int(s["blocks"]),
                     "peak MB": s["peak_mb"],
                     "R0 (low - high)": f"{n_(lo)} - {n_(hi)}",
                     "R3": int(fo["n_r3"].sum()), "kept": int(fo["n_kept"].sum()),
                     "dest. served": f"{int((fo['n_kept'] > 0).sum())} / "
                                     f"{int(fo['pt'].sum())}"})
        cost.append({"origin": f"{name_of(x, o)} ({o})",
                     "3-link chains": int(s["chains3"]),
                     "skipped p1": pct(s["skipped1"], s["chains3"]),
                     "skipped p2": pct(s["skipped2"], s["chains3"]),
                     "triple calls p1": int(s["calls1"]), "triple calls p2": int(s["calls2"]),
                     "reused p2": int(s["reused2"]), "exact R0 (p2)": int(s["evaluated2"]),
                     "R0 by bounds": int(s["r0_by_bounds"]),
                     "calls per s": s["calls_per_s"]})
    tot = pd.DataFrame(rows)
    sub = tt.loc[origins]
    stage_c_rec = record_of(x["records"], "enumerate", origins)
    fc = f[f["o"].isin(origins)]
    sums = {"r0": int(fc["n_r0"].sum()), "r0_bound": int(fc["n_r0_bound"].sum()),
            "r3": int(fc["n_r3"].sum()),
            "triple_calls": int((sub["calls1"] + sub["calls2"] + sub["calls_trace"]).sum()),
            "kept": int(fc["n_kept"].sum())}
    if stage_c_rec:
        ct = stage_c_rec["info"].get("totals", {})
        same = [k for k in ("r0", "r0_bound", "r3", "triple_calls")
                if k in ct and int(ct[k]) == sums[k]]
        differ = [k for k in ("r0", "r0_bound", "r3", "triple_calls")
                  if k in ct and int(ct[k]) != sums[k]]
        pk = {int(k): float(v) for k, v in ct.get("peak_mb", {}).items()}
        big = max(pk, key=pk.get) if pk else None
        c_text = (
            f"The separate stage C run of {stage_c_rec['time']} (ps-manifest, info totals) "
            f"ran under R2_GATE {stage_c_rec['params'].get('R2_GATE')} with "
            f"{stage_c_rec['info']['workers']} workers in "
            f"{float(stage_c_rec['info']['seconds']):,.0f} s. The search does not depend on "
            "R2_GATE, and its totals over these ten origins equal stage D's for "
            + (", ".join(same) if same else "none of r0, r0_bound, r3, triple_calls")
            + (f" (they DIFFER for {', '.join(differ)})" if differ else "")
            + f": r0 {n_(sums['r0'])}, r0_bound {n_(sums['r0_bound'])}, r3 "
            f"{n_(sums['r3'])}, triple calls {n_(sums['triple_calls'])} (passes 1, 2 and the "
            "trace search). So the R0, R3, chain and call columns below are the stage C "
            "figures. The kept column, the seconds and the memory are stage D's: stage C "
            f"kept {n_(ct.get('kept'))} (stage D {n_(sums['kept'])} under "
            f"{p.get('R2_GATE')}), and its peaks differ"
            + (f" (e.g. {name_of(x, big)} {pk[big]:,.1f} MB in stage C against "
               f"{float(tt.loc[big, 'peak_mb']):,.1f} MB here)" if big is not None else "")
            + ".")
    else:
        c_text = "No manifest record of a separate stage C run was found."
    lines = ["## 2. Stage C: ten origins", "",
             "The ten origins of STAGE_C_ORIGINS, as they ran inside stage D. Seconds and "
             "memory come from the attrs of each funnel shard (attrs['search'], "
             "search.run_origin); counts from ps-funnel.", "", c_text, "",
             md(tot, 1), "",
             "total: the whole origin in its worker (pass 1 minima, pass 2 counts and "
             "keeps, the exact rows and tables, the rules); trace: the extra search at "
             "MU = inf of the named pairs (origins "
             + ", ".join(str(o) for o in origins if o in named_o)
             + "); rows etc.: total less the passes and the rules. peak MB: the peak "
             "working set of the worker process up to the end of this origin "
             "(search.peak_mb; it covers the network and every earlier origin of the same "
             "worker, so it never falls within a worker). R0: the interval of section 5 "
             "(ps-funnel n_r0 - n_r0_bound to n_r0; the named destinations are exact). "
             "dest. served: destinations with any kept alternative / with a kept service "
             "(non-car) alternative, of 383.", "",
             "Cost of counting chains date by date. Two-link chains are counted once, "
             "before the search, in pathset-transfers.py"
             + (f": {n_(ctx['transfers']['admissible'])} admissible link pairs, "
                f"{n_(ctx['transfers']['rows'])} with an opportunity, "
                f"{ctx['transfers']['pairs_s']:.1f} s for the counting "
                f"(ps-manifest, transfers record of {ctx['transfers']['time']}, "
                "info seconds.pairs_s)" if ctx.get("transfers") else "")
             + ". Three-link chains are counted inside the search "
             "(transfer.triple_stats, date by date when the links share at least "
             f"MIN_COMMON_DATES = {p['MIN_COMMON_DATES']} dates, else on typical days, "
             "always on typical days when a transfer joins air and ground). A chain is "
             "evaluated only when its lower bounds can matter (pass 1: below a current "
             "minimum; pass 2: inside the R3 margin); results are cached between the "
             "passes.", "",
             "What this does not measure: the search does not count date-by-date and "
             "typical-day evaluations apart (attrs['search'] has no counter by basis; "
             "search.STAT_NAMES). So the cost of date-by-date counting on its own, which "
             "the plan asks for, is not known; the table gives the cost of all "
             "three-link evaluations. Counting triple_stats calls by basis in a later run "
             "would answer it.", "",
             md(pd.DataFrame(cost), 0), "",
             "3-link chains: chains that passed the prefix clauses of R0 (pass 2; the "
             "same in pass 1). skipped: share not evaluated (bounds outside what the pass "
             "needs, or no destination left). triple calls: evaluations "
             "(cache misses); reused: cache hits in pass 2. exact R0: R0 candidates "
             "with exact values in pass 2 (one and two links always, three links when "
             "evaluated). R0 by bounds: three-link candidates counted without "
             "evaluation (section 5). calls per s: calls of passes 1 and 2 over their "
             "seconds; the passes also walk the chains, so the calls alone run faster.",
             "",
             f"Over the ten origins: {n_(sub['calls1'].sum() + sub['calls2'].sum())} "
             f"triple calls in {sub['pass1_s'].sum() + sub['pass2_s'].sum():,.0f} s of passes "
             f"({(sub['calls1'].sum() + sub['calls2'].sum()) / (sub['pass1_s'].sum() + sub['pass2_s'].sum()):,.0f} "
             f"calls per second of pass time); pass 2 skipped "
             f"{pct(sub['skipped2'].sum(), sub['chains3'].sum())} of "
             f"{n_(sub['chains3'].sum())} three-link chains. The search, not the rules, "
             f"takes the time: rules {sub['rules_s'].sum():,.0f} s of "
             f"{sub['total_s'].sum():,.0f} s. "
             + cache_sentence(sub.reset_index(), x), ""]
    return "\n".join(lines)


def cache_floor(attrs: dict, origin: int, budget_gb: float) -> float:
    """Least number of entries the triple cache held when pass 2 of
    ``origin`` emptied it: search._srch_triple empties it when (entries + 1)
    x CACHE_BYTES plus the candidate rows written so far (written x ID_BYTES
    + triple_rows x TV_BYTES) exceed MEM_CAND_GB; the rows written by the
    end of the pass are an upper bound of those written at that moment."""
    p2 = attrs[origin]["search"]["pass2"]
    held = int(p2["written"]) * search.ID_BYTES + int(p2["triple_rows"]) * search.TV_BYTES
    return (float(budget_gb) * 2 ** 30 - held) / search.CACHE_BYTES


def cache_sentence(tt: pd.DataFrame, x: dict) -> str:
    """How often the triple cache was emptied (counter cache_cleared), and
    the largest size recorded (at the end of a pass)."""
    c = tt[tt["cache_cleared"] > 0]
    i = tt["cache_max"].idxmax()
    big = (f"the largest size recorded, at the end of a pass, is {n_(tt.loc[i, 'cache_max'])} "
           f"entries ({name_of(x, tt.loc[i, 'origin'])})")
    if not len(c):
        return f"The triple cache was never emptied; {big}."
    def times(k):
        return "once" if int(k) == 1 else f"{int(k)} times"
    floor = []
    for o in c["origin"]:
        try:
            floor.append(f"{name_of(x, o)} at least "
                         f"{cache_floor(x['attrs'], int(o), x['params']['MEM_CAND_GB']) / 1e6:.1f} "
                         "million entries")
        except (KeyError, TypeError):
            pass
    return ("The triple cache was emptied when it reached the memory budget "
            f"(cache_cleared): {', '.join(f'{name_of(x, o)} ({int(o)}) {times(k)}' for o, k in zip(c['origin'], c['cache_cleared']))}. "
            f"The size is recorded at the end of each pass only: {big}. A cache that was "
            "emptied held more just before"
            + (f" ({'; '.join(floor)}, by the budget rule of search._srch_triple)"
               if floor else "")
            + ". Emptying never changes a result (search.py docstring), only the time.")


def section_stage_d(x: dict, ctx: dict) -> str:
    tt = ctx["timing"]
    run = x["run"]
    log = x["log"]
    info = run["info"]
    wall = (log["seconds"] if log and log.get("seconds") is not None
            else float(info["seconds"]))
    src = ("console log of the run (first line 'enumerate: ...' to the last "
           "'saved paths/funnel/...')" if log and log.get("seconds") is not None
           else "ps-manifest record, info seconds")
    tot = tt["total_s"].sum()
    q_s, q_m = quantiles(tt["total_s"]), quantiles(tt["peak_mb"])
    slow = tt.sort_values("total_s", ascending=False).head(10)
    slow_rows = pd.DataFrame({
        "origin": [f"{name_of(x, o)} ({o})" for o in slow["origin"]],
        "total s": slow["total_s"].to_numpy(), "pass 1 s": slow["pass1_s"].to_numpy(),
        "pass 2 s": slow["pass2_s"].to_numpy(), "rules s": slow["rules_s"].to_numpy(),
        "3-link chains": slow["chains3"].astype(int).to_numpy(),
        "triple calls": (slow["calls1"] + slow["calls2"]).astype(int).to_numpy(),
        "peak MB": slow["peak_mb"].to_numpy()})
    traced = slow[slow["calls_trace"] > 0]
    multi = tt[tt["blocks"] > 1]
    first4 = tt[tt["origin"].isin(sorted(info["origins_run"])[:int(info["workers"])])]
    machine = ctx["machine"]
    lines = ["## 3. Stage D: the run", "",
             f"Wall clock {wall_text(wall)} ({wall:,.0f} s) from the {src}"
             + (f"; the manifest record gives {float(info['seconds']):,.1f} s" if log else "")
             + f". The sum of the per-origin seconds (attrs total_s) is {tot:,.0f} s, "
             f"{ctx['use']:.0%} of {info['workers']} workers x the wall clock.", "",
             f"Machine: {machine}. {info['workers']} worker processes "
             f"(WORKERS {x['params'].get('WORKERS')}), each with NUMBA_NUM_THREADS = 1 "
             "(search.run_pool), each building the network once.", "",
             md(pd.DataFrame([
                 ("seconds per origin (total_s)", tt["total_s"].median(), q_s["p90"], q_s["max"]),
                 ("peak MB of the worker after the origin (peak_mb)", tt["peak_mb"].median(),
                  q_m["p90"], q_m["max"]),
                 ("resident MB when the origin started (rss_start_mb)",
                  tt["rss_start_mb"].median(), quantiles(tt["rss_start_mb"])["p90"],
                  tt["rss_start_mb"].max())],
                 columns=["", "median", "p90", "max"]), 0), "",
             "peak_mb only grows within a worker (it covers the network and every "
             "origin before), so its median is not the need of a typical origin; its "
             f"maximum, {q_m['max']:,.0f} MB, is the largest working set any worker "
             "reached. The first origin of each worker includes the compilation of the "
             "search kernels in pass 1. search.run_pool queues the origins in id order, so "
             f"the first {info['workers']} ids are the first origins of the workers: "
             + ", ".join(f"{name_of(x, o)} ({o}) pass 1 {s:.0f} s"
                         for o, s in zip(first4["origin"], first4["pass1_s"]))
             + f" (median pass 1 of all origins {tt['pass1_s'].median():.0f} s).", "",
             "The ten slowest origins (triple calls: passes 1 and 2, as in section 2"
             + ("; the trace search of the named pairs adds "
                + ", ".join(f"{n_(c)} for {name_of(x, o)}"
                            for o, c in zip(traced["origin"], traced["calls_trace"]))
                if len(traced) else "") + "):", "", md(slow_rows, 1), "",
             (f"Origins done in more than one destination block (MEM_CAND_GB "
              f"{x['params'].get('MEM_CAND_GB')}): "
              + ", ".join(f"{name_of(x, o)} ({o}, {b} blocks, buffer {m:,.0f} MB)"
                          for o, b, m in zip(multi["origin"], multi["blocks"], multi["buffer_mb"]))
              if len(multi) else "No origin needed more than one destination block.")
             + "", ""]
    return "\n".join(lines)


FUNNEL_MEANING = [
    ("n_r0", "R0 candidates: paths of one to three service links (plus car access and "
     "egress) that pass every clause of R0 and exist, plus the car where ps-car has "
     "a row; three-link chains cut by R3 on their bounds are counted without "
     "evaluation (section 5)"),
    ("n_r0_bound", "the part of n_r0 counted on bounds (0 for the named pairs, whose "
     "n_r0 comes from the trace search at MU = inf)"),
    ("n_r3", "candidates R3 keeps (gtt or gtc within (1 + MU) x the least of the mode "
     "class), car included"),
    ("n_r1", "alternatives after R1: groups of one mode sequence whose T lie within "
     "eps_T of a leader; these enter R2"),
    ("n_r2", "alternatives left after R2 (and the comparison across classes, off)"),
    ("n_kept", "alternatives kept after the cap R5 = n_r2 - n_capped"),
    ("n_capped", "alternatives the cap dropped"),
]


def section_funnel(x: dict, ctx: dict) -> str:
    f = x["funnel"]
    lo, hi = r0_interval(f["n_r0"].sum(), f["n_r0_bound"].sum())
    tot = [("n_r0", f["n_r0"].sum()), ("n_r0_bound", f["n_r0_bound"].sum()),
           ("exact R0, lower end", lo), ("n_r3", f["n_r3"].sum()), ("n_r1", f["n_r1"].sum()),
           ("n_r2", f["n_r2"].sum()), ("n_kept", f["n_kept"].sum()),
           ("n_capped", f["n_capped"].sum())]
    q = f["n_kept"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    lines = ["## 4. The funnel over all ordered pairs", "",
             f"ps-funnel holds {len(f):,} ordered pairs ({f['o'].nunique()} origins x "
             f"{f['o'].nunique() - 1} destinations). Totals:", "",
             md(pd.DataFrame([(k, int(v)) for k, v in tot], columns=["column", "total"])), "",
             md(pd.DataFrame(FUNNEL_MEANING, columns=["column", "what it counts "
                                                      "(search.run_origin)"])), "",
             f"From R0 to the kept set: the exact R0 count lies between {n_(lo)} and "
             f"{n_(hi)}; R3 keeps {n_(f['n_r3'].sum())} ({pct(f['n_r3'].sum(), hi)} of "
             f"the upper end, {pct(f['n_r3'].sum(), lo)} of the lower); R1 turns them into "
             f"{n_(f['n_r1'].sum())} alternatives; R2 leaves {n_(f['n_r2'].sum())}; the cap "
             f"drops {n_(f['n_capped'].sum())} in {n_((f['n_capped'] > 0).sum())} pairs and "
             f"{n_(f['n_kept'].sum())} are kept (n_r2 = n_kept + n_capped by definition: "
             "search.run_origin writes n_r2 as their sum).", "",
             "Kept alternatives per pair (ps-funnel n_kept): "
             + ", ".join(f"{k} {v:g}" for k, v in zip(
                 ["min", "p10", "p25", "median", "p75", "p90", "max"], q.to_numpy()))
             + f"; mean {f['n_kept'].mean():.2f}. Every pair has at least one alternative "
             f"({int((f['n_kept'] == 0).sum())} pairs without).", ""]
    return "\n".join(lines)


def section_sequences(x: dict, ctx: dict) -> str:
    p = x["params"]
    st = seq_table(x["seq_counts"])
    tt = ctx["timing"]
    named_o = {int(a) for a, _ in p["NAMED_PAIRS"]}
    f = x["funnel"]
    chk = []
    for o in tt["origin"]:
        if o in named_o:
            continue
        fo = f[f["o"] == o]
        lo = int((fo["n_r0"] - fo["n_r0_bound"]).sum())
        ev = int(tt.loc[tt["origin"] == o, "evaluated2"].iloc[0])
        chk.append(lo == ev + int(np.isfinite(fo["car_min"].to_numpy(float)).sum()))
    lines = ["## 5. Counts by mode sequence and the deck's Table 1", "",
             "How to read n_r0 for three links (checked in pathset/search.py, "
             "_srch_ends, pass 2). A three-link chain is evaluated "
             "(transfer.triple_stats) only when, for at least one destination its last "
             "hub serves, its lower bounds can lie inside the R3 margin. Otherwise it is "
             "counted as an R0 candidate of every destination that passes the other R0 "
             "clauses and whose LOWER BOUND of T is at most T_MAX_MIN, and is added to "
             "n_r0 and n_r0_bound without evaluation.", "",
             "Two things are then not checked: that the chain exists (an opportunity on at "
             f"least MIN_ACTIVE_DATES = {p['MIN_ACTIVE_DATES']} dates when counted date by "
             "date, or on at least one weekday on typical days; _srch_exists), and that "
             "its exact T is at most T_MAX_MIN. Such a candidate would be cut by R3 if it "
             "exists, so the kept set is not affected, only the count. One- and two-link "
             "candidates are always evaluated (n_r0_bound = 0 for every one- and two-link "
             "sequence below). So for three-link sequences the exact count lies in "
             "[n_r0 - n_r0_bound, n_r0].", "",
             "Data check: for every origin without named pairs, n_r0 - n_r0_bound summed "
             "over its destinations equals the candidates pass 2 evaluated (attrs "
             f"'evaluated') plus its car candidates ({sum(chk)} of {len(chk)} origins). For "
             "the named pairs the run replaced n_r0 by the exact count of the trace "
             "search (n_r0_bound 0 there).", "",
             "All mode sequences (ps-seq-counts summed over origins; C = car):", ""]
    tab = st.copy()
    tab["R0"] = [n_(h) if e else f"{n_(l)} - {n_(h)}"
                 for l, h, e in zip(tab["r0_low"], tab["r0_high"], tab["exact"])]
    lines += [md(tab[["seq", "links", "R0", "n_r3", "n_kept"]]
                 .rename(columns={"R0": "R0 (exact, or low - high)", "n_r3": "R3",
                                  "n_kept": "kept"})), ""]
    three = st[st["links"] == 3]
    lines += [f"Three-link sequences: between {n_(three['r0_low'].sum())} and "
              f"{n_(three['r0_high'].sum())} R0 candidates; R3 keeps "
              f"{n_(three['n_r3'].sum())}; {n_(three['n_kept'].sum())} are kept "
              f"({pct(three['n_kept'].sum(), st['n_kept'].sum())} of all kept).", ""]
    # Table 1 side by side
    seqs = DECK_TABLE1["seq"].tolist()
    mb = ctx["metrics_best"]
    mk = ctx["metrics_kept"]
    rows = []
    for _, d in DECK_TABLE1.iterrows():
        s = d["seq"]
        r = st[st["seq"] == s].iloc[0]
        b = mb[mb["seq"] == s].iloc[0]
        k = mk[mk["seq"] == s].iloc[0]
        def cell(k):
            return f"{num(d[f'{k}_mean'])} ({num(d[f'{k}_sd'])})"
        rows.append({"seq": s, "set": "deck: all paths", "n": n_(d["paths"]),
                     "access": cell("acc"), "egress": cell("egr"), "IVT air": cell("ride_A"),
                     "IVT bus": cell("ride_B"), "IVT rail": cell("ride_R"), "km": cell("km")})
        rows.append({"seq": s, "set": "ours: R0 count", "n": n_(r["n_r0"]), "access": "",
                     "egress": "", "IVT air": "", "IVT bus": "", "IVT rail": "", "km": ""})
        for label, m in (("ours: best per pair", b), ("ours: kept", k)):
            rows.append({"seq": s, "set": label, "n": n_(m["rows"]), "access": m["acc"],
                         "egress": m["egr"], "IVT air": m["ride_A"], "IVT bus": m["ride_B"],
                         "IVT rail": m["ride_R"], "km": m["km"]})
    lines += ["The twelve sequences of the deck's Table 1 (slide 223, 'Path counts and mean "
              "(standard deviation) by mode sequence'). Deck: its count of paths and "
              "mean (sd). Ours: the R0 count (exact for one and two links); then mean "
              "(sd) over ps-best-by-sequence (one row per FUA pair with an R0 candidate "
              "of the sequence: the one with the least gtt, before R3) and over the kept "
              "alternatives of the sequence (ps-alternatives). access, egress: acc and "
              "egr, clock minutes of the car legs (unweighted; t_a and t_e are the "
              "weighted ones). IVT: unweighted riding minutes of the path's links of "
              "that mode (ps-links ride, the date-weighted mean of the link's riding-time "
              "class); t_ivt of ps-alternatives is the weighted sum and is not used. km: "
              "sum over the path's links of ps-links km, the great-circle distance "
              "between the link's board and alight hub (pathset/network.py, gc_km of "
              "the hub coordinates: an airport's own, a station hub's mean of its "
              "stations); access and egress are not included.", "",
              md(pd.DataFrame(rows)), ""]
    lines += section_definitions(x, ctx)
    return "\n".join(lines)


def section_definitions(x: dict, ctx: dict) -> list:
    """The definitional differences between the deck's counts and ours."""
    st = seq_table(x["seq_counts"]).set_index("seq")
    hp = ctx["hub_pairs"]
    d1 = DECK_TABLE1.set_index("seq")
    acc = ctx["airport_access"]
    per = ctx["link_periods"]
    out = ["What differs in definition, and so explains gaps between the two columns. "
           "Marked FACT where code, data or the deck shows it, POSSIBLE where it is a "
           "likely reason I have not proven:", ""]
    out.append(
        f"- FACT, link key. The deck's service link is \"{DECK_QUOTES[232][0]}\" "
        "(slide 232). Ours is (board hub, alight hub, mode, riding-time class) with the "
        "operators pooled (plan, Definitions, 'Service link'; ps-link-members gives "
        f"the way back). ps-links has {n_(hp['R']['links'])} rail links on "
        f"{n_(hp['R']['pairs'])} ordered hub pairs ({hp['R']['links'] / hp['R']['pairs']:.1f} "
        f"riding-time classes per pair, up to {hp['R']['max']}), "
        f"{n_(hp['B']['links'])} coach links on {n_(hp['B']['pairs'])} hub pairs "
        f"({hp['B']['links'] / hp['B']['pairs']:.1f} per pair) and one air link per "
        f"airport pair ({n_(hp['A']['links'])}). Our direct counts are R "
        f"{n_(st.loc['R', 'n_r0'])} and B {n_(st.loc['B', 'n_r0'])}, "
        f"{st.loc['R', 'n_r0'] / d1.loc['R', 'paths']:.1f} and "
        f"{st.loc['B', 'n_r0'] / d1.loc['B', 'paths']:.1f} times the deck's; the numbers "
        f"of ordered hub pairs ({n_(hp['R']['pairs'])} rail, {n_(hp['B']['pairs'])} "
        f"coach) are close to the deck's direct counts ({n_(d1.loc['R', 'paths'])} and "
        f"{n_(d1.loc['B', 'paths'])}). POSSIBLE: the riding-time classes explain most of "
        "the gap for direct ground paths; the station sets also differ (next point), "
        "so the closeness may be partly chance.")
    out.append(
        "- FACT, which hubs a FUA reaches. A FUA reaches only the station hubs inside "
        "it (connect.station_connectors: one connector per station hub, to its own FUA; "
        f"{n_(ctx['n_station_hubs'])} station hubs in {ctx['n_station_fuas']} FUAs) and "
        "the airports whose catchment list holds it, when the drive is at most "
        f"ACCESS_MAX_AIR_MIN = {x['params'].get('ACCESS_MAX_AIR_MIN')} min "
        f"(connect.airport_connectors; {n_(acc['n'])} airport access rows, mean "
        f"{acc['mean']:.0f} min). The deck's mean access for A is "
        f"{d1.loc['A', 'acc_mean']} min with sd {d1.loc['A', 'acc_sd']}. With every leg "
        "between 0 and 120 min that is only just possible (nearly all legs at 0 or at 120 "
        "min), so many of its airport access legs are probably longer than 120 min; ours "
        f"cannot be. Our A count ({n_(st.loc['A', 'n_r0'])}) is "
        f"{pct(st.loc['A', 'n_r0'], d1.loc['A', 'paths'], 0)} of the deck's. POSSIBLE: "
        "the access cap, together with a different airport list, explains most of that "
        "gap and our lower access means ("
        f"{x['best'].loc[x['best']['seq'] == 'A', 'acc'].mean():.0f} min over "
        f"ps-best-by-sequence against {d1.loc['A', 'acc_mean']}). Rail and "
        "bus come from data-upstream-hubs (plan, decision 2), not from the feeds of the "
        "deck's 3MG, so the station sets differ.")
    out.append(
        f"- FACT, dates. The deck samples \"{DECK_QUOTES[232][1]}\" (slide 232: "
        f"\"{DECK_QUOTES[232][2]}\"). We use every date from SERVICE_START to SERVICE_END "
        f"({x['params'].get('SERVICE_START')} to {x['params'].get('SERVICE_END')}) and keep "
        f"links active on at least MIN_ACTIVE_DATES = {x['params'].get('MIN_ACTIVE_DATES')} "
        "dates. In ps-links all "
        f"{n_(per['A']['n'])} air links end before 2026 (last_day), while "
        f"{n_(per['B']['only_2026'])} of {n_(per['B']['n'])} coach links and "
        f"{n_(per['R']['only_2026'])} of {n_(per['R']['n'])} rail links run only in 2026. "
        "POSSIBLE: a sample of 2023-2025 days misses services that exist only in "
        "later feeds and the reverse; I cannot tell how the deck's feeds are dated.")
    out.append(
        "- FACT, R0 and existence. Our candidates pass R0 (search.py: stops at distinct "
        "places, transfer cities neither o, d nor the first or last city and distinct, "
        f"no link passing o, d or a city of the path on at least PASSED_SHARE = "
        f"{x['params'].get('PASSED_SHARE')} of its dated departures, one air block, T <= "
        f"{x['params'].get('T_MAX_MIN')} min), and a two-link chain must exist: an "
        "opportunity inside the transfer window on at least 5 dates, or on one typical "
        "weekday. The deck's slides 222-223 do not say whether a one-transfer path "
        f"needed a connection inside a window (slide 222: \"{DECK_QUOTES[222][0]}\", "
        f"\"{DECK_QUOTES[222][1]}\"); the windows appear later (slide 234).")
    rt = one_transfer_ratios(st)
    big = rt[rt["ratio"] > 1].sort_values("ratio")
    small = rt[rt["ratio"] <= 1].sort_values("ratio")
    rail_c = hp["R"]["links"] / hp["R"]["pairs"]
    bus_c = hp["B"]["links"] / hp["B"]["pairs"]
    acc_ratio = d1.loc["A", "paths"] / st.loc["A", "n_r0"]
    moves = ctx.get("air_station_moves", {})
    out.append(
        "- FACT, the direction of the one-transfer gaps (our R0 count / the deck's count): "
        + ", ".join(f"{s} {ratio_text(r)}" for s, r in zip(rt["seq"], rt["ratio"])) + ". The deck "
        "is larger only for " + " and ".join(small["seq"]) + "; ours is larger for the "
        f"other {len(big)}, from {big['ratio'].min():.1f} to {big['ratio'].max():.0f} times. "
        "One explanation cannot cover both directions. POSSIBLE causes where the deck is "
        f"larger: airport access beyond 120 min (the direct A count is already "
        f"{acc_ratio:.1f} times ours, previous points), and graph paths without a "
        "connection test if the deck counted those. POSSIBLE causes where ours is "
        f"larger: riding-time classes multiply over two links ({rail_c:.1f} rail and "
        f"{bus_c:.1f} coach classes per hub pair); for air-ground sequences, our "
        "matching on typical weekdays across years (next point) and our "
        + (f"{n_(moves.get('n'))} airport-to-station moves ({n_(moves.get('road'))} by "
           f"road, {n_(moves.get('colocated'))} co-located; ps-moves)"
           if moves else "airport-to-station moves (ps-moves)")
        + ", where slide 219 says 3MG had no connection between airports and stations "
        "before its 21 Oct changes. None of these is proven; the riding-time classes "
        f"alone would make BB larger here, yet the deck's BB is "
        f"{1 / float(rt.set_index('seq').loc['BB', 'ratio']):.1f} times ours.")
    out.append(
        "- FACT, typical days for air and ground. A chain that joins air and ground is "
        "counted on typical weekdays only (transfer.py, 'Basis'; search._srch_triple, "
        "'forced'); with air data ending in 2025 and most ground data in 2026 it could "
        "not be counted on dates (section 8). The deck finds almost no air-ground "
        "connections on its 2023-2025 sample (plan, Verification 2), and the bar chart "
        "of slide 234 shows almost no feasible AR, BA or RA connections. Its Air1 counts "
        f"({n_(d1.loc[['AB', 'AR', 'BA', 'RA'], 'paths'].min())} to "
        f"{n_(d1.loc[['AB', 'AR', 'BA', 'RA'], 'paths'].max())} for AB, AR, BA, RA) are "
        f"{rt.set_index('seq').loc[['AB', 'AR', 'BA', 'RA'], 'ratio'].min():.0f} to "
        f"{rt.set_index('seq').loc[['AB', 'AR', 'BA', 'RA'], 'ratio'].max():.0f} times "
        "smaller than our R0 counts. POSSIBLE: the deck counted only connections that "
        "exist on its sampled dates, where air and ground rarely overlap; the slides do "
        "not say how these counts were made.")
    out.append(
        "- FACT, selection. Our means are over the least-gtt candidate per pair "
        "(ps-best-by-sequence) and over the kept alternatives, both short and fast by "
        "construction; the deck's are over all its paths. The IVT columns compare "
        "unweighted minutes: the deck's look unweighted too (A: 1,197 km in 115 min "
        "is 625 km/h; weighted by 1.56 the riding time would mean about 1,000 km/h), "
        "but the slide does not say. Length: slide 219 says link lengths were added, "
        "great-circle distance for air links; how ground links are measured is not "
        "stated.")
    out.append(
        "- FACT, scope. Our set has one more sequence (RAR: one air block, at most two "
        "flights, 36 words; the deck's path set has 35, slide 238) and the car (C). "
        "Neither is in Table 1.")
    out.append("")
    return out


def section_sets(x: dict, ctx: dict) -> str:
    f, p = x["funnel"], x["params"]
    tab = set_size_table(f, p)
    dist = size_distribution(f)
    ex = class_extras(x["alts"])
    car_pairs = int((x["alts"]["seq"].astype(str) == "C").sum())
    tr = x["trace"]
    capped_tr = tr[tr["status"].astype(str) == "capped"]
    lines = ["## 6. Set sizes and mode classes", "",
             f"Mode classes are the sets of modes a path uses (A, AB, ABR, AR, B, BR, R) "
             f"and the car (C): at most 8. The cap of a pair is max(K_CAP {p['K_CAP']}, "
             f"classes present + K_EXTRA {p['K_EXTRA']}), so 9 to 11 when 6 to 8 classes "
             "are present. Pairs by the number of alternatives kept (ps-funnel n_kept):", "",
             md(pd.DataFrame([["pairs"] + dist["pairs"].astype(int).tolist()],
                             columns=["alternatives"] + [str(v) for v in dist["alternatives"]])),
             "",
             "By the number of classes kept (ps-funnel n_classes; with CROSS_CLASS off "
             "and R4 on every class of the R3 candidates keeps an alternative, so this is "
             "also the number of classes present):", "",
             md(tab.rename(columns={"n_classes": "classes", "at_cap": "pairs at cap",
                                    "pairs_capped": "pairs capped",
                                    "capped": "alternatives capped"}), 2), "",
             f"The cap acts in {n_((f['n_capped'] > 0).sum())} of {len(f):,} pairs "
             f"({pct((f['n_capped'] > 0).sum(), len(f))}) and drops "
             f"{n_(f['n_capped'].sum())} alternatives. In every capped pair the kept "
             f"set has exactly the cap ({ctx['capped_not_at_cap']} exceptions), so in "
             "these pairs the set size is the cap's, not the rules'. The classes of the "
             "dropped alternatives are not stored for the full run (ps-funnel has only "
             "the count; dropped alternatives are not in ps-alternatives). ps-trace has "
             f"them for the named pairs: {len(capped_tr)} dropped, by class "
             + ", ".join(f"{k} {v}" for k, v in capped_tr["cls"].value_counts().items())
             + ". " + cap_lines(x), "",
             f"One alternative per class against extra ones. The {n_(ex['alternatives'])} "
             f"kept alternatives fill {n_(ex['cells'])} (pair, class) cells: one per class "
             f"present, and {n_(ex['extras'])} more ({pct(ex['extras'], ex['alternatives'])} "
             "of the set) are second or later alternatives of a class. R4 keeps the last "
             "alternative of every class against the cap; CROSS_CLASS is off, so no "
             "class removes another. Per class: cells (pairs where the class is kept), "
             "cells with more than one alternative, alternatives:", "",
             md(ex["per_class"].rename(columns={"cls": "class", "multi": "cells with 2+"})),
             "",
             f"The car is one of the classes in {n_(car_pairs)} pairs. Against the "
             f"literature range the plan quotes ({PLAN_LITERATURE}): our median set "
             f"({f['n_kept'].median():g}) and mean ({f['n_kept'].mean():.1f}) lie inside "
             "the generated range and well above the observed one. Most of the size "
             "comes from one alternative per mode class; a choice set of 2-3 observed "
             "alternatives per traveller is a different object from a master set per "
             "FUA pair over all modes, so the comparison is loose.", ""]
    dist = ctx.get("distance")
    if dist is not None:
        short = dist.iloc[0]
        lines += [
            "By distance between the FUA centres (great circle, fuas.centre), set sizes fall "
            "as the distance grows: short pairs have the largest sets, and many of their "
            "alternatives are detours (ps-funnel, ps-alternatives; '2 x least T or more': "
            "kept service alternatives whose T is at least twice the least T of the pair's "
            "kept service alternatives; 'with a flight': those that fly):", "",
            md(dist, 1), "",
            f"Under {DISTANCE_BANDS[1]} km, {short['2 x least T or more']} of the kept service "
            f"alternatives take at least twice the best service time and "
            f"{short['with a flight']} fly. One alternative per mode class (R4) fills short "
            f"pairs with such paths (item {item_no('distance')}).", ""]
    return "\n".join(lines)


def cap_lines(x: dict) -> str:
    """The cap on the sensitivity origins (no_cap against default)."""
    sa = x.get("sens_alts")
    if sa is None:
        return ("For Wien and Katowice the sensitivity run would give them (no_cap "
                "variant); its tables are missing.")
    out = []
    for o in sa["origin"].unique():
        a = sa[sa["origin"] == o]
        c = cap_detail(a[a["variant"] == "default"], a[a["variant"] == "no_cap"], x["params"])
        out.append(
            f"{name_of(x, o)} (sensitivity run, no_cap against default): the cap dropped "
            f"{n_(c['dropped'])} alternatives, by class "
            + ", ".join(f"{k} {v}" for k, v in c["dropped_by_class"].items())
            + f"; {n_(c['protected'])} kept alternatives in {n_(c['pairs_protected'])} pairs "
            "lie beyond the cap in gtc order and stay only as the last of their class (R4), "
            "by class " + (", ".join(f"{k} {v}" for k, v in c["protected_by_class"].items())
                           or "none") + ".")
    return " ".join(out)


def section_unserved(x: dict, ctx: dict) -> str:
    u = x["unserved"]
    rows = []
    reach = ctx["reach"]
    conn, hubs = x["connectors"], x["hubs"].set_index("hub")
    for o, d, reason in zip(u["o"], u["d"], u["reason"]):
        def ends(fua, col):
            c = conn[(conn["fua"] == fua) & np.isfinite(conn[col].to_numpy(np.float64))]
            k = hubs.loc[c["hub"], "kind"].value_counts()
            return f"{int(k.get('S', 0))} S, {int(k.get('A', 0))} A"
        lv = reach[(int(o), int(d))]
        why = ("needs 4 or more services" if sum(lv) == 0 else
               ("3-service chains fail R0, the timetable or 24 h" if lv[0] + lv[1] == 0
                else "a 2-service chain fails R0, the timetable or 24 h"))
        rows.append({"origin": f"{name_of(x, o)} ({o})", "destination": f"{name_of(x, d)} ({d})",
                     "km": int(round(float(pair_km([int(o)], [int(d)], ctx["centres"])[0]))),
                     "origin hubs": ends(o, "acc_min"), "dest. hubs": ends(d, "egr_min"),
                     "graph chains 1/2/3": "/".join(str(v) for v in lv),
                     "likely reason": why})
    tab = pd.DataFrame(rows)
    fuas_u = sorted(set(u["o"]) | set(u["d"]))
    cov = ctx["car_cov"]
    groups = cov.groupby("out")["fua"].apply(list).sort_index()
    f = x["funnel"]
    km_all = pair_km(f["o"].to_numpy(), f["d"].to_numpy(), ctx["centres"])
    lines = ["## 7. Unserved pairs and pairs without a car", "",
             f"ps-unserved holds {len(u)} ordered pairs, all with reason "
             + ", ".join(f"'{k}' ({v})" for k, v in u["reason"].value_counts().items())
             + ": the car is their only alternative; no pair is without any alternative. "
             f"They involve {len(fuas_u)} FUAs: "
             + ", ".join(f"{name_of(x, i)} ({i})" for i in fuas_u)
             + "; by country (fuas.icc): "
             + ", ".join(f"{k} {v}" for k, v in x["fuas"].set_index("id")["icc"]
                         .reindex(fuas_u).value_counts().items())
             + f". The pairs lie {tab['km'].min():,} to {tab['km'].max():,} km apart "
             f"(FUA centres, great circle; median {tab['km'].median():,.0f} km, against "
             f"{np.nanmedian(km_all):,.0f} km over all pairs). The hubs each end can use "
             "are in the table; some ends have several (I did not measure how many links "
             "leave them).", "",
             md(tab), "",
             "origin hubs / dest. hubs: station (S) and airport (A) hubs with a finite "
             "access (origin) or egress (destination) in ps-connectors. graph chains: in "
             "the graph of ps-transfers rows that exist (basis D with n_days >= 5, or T), "
             "the number of distinct last links that end at a hub of the destination "
             "after 1, 2, 3 services from the hubs of the origin, without R0 clauses, "
             "timetable or 24-hour limit. So a 0/0/0 pair needs four or more services; "
             "for the others the search found that no chain of three passes R0, exists "
             "(three-link existence is decided by transfer.triple_stats, not by the "
             "graph) and arrives within T_MAX_MIN. I have not separated these three "
             "causes per pair.", "",
             f"Pairs without a car alternative: {n_(f['car_min'].isna().sum())} (ps-funnel "
             "car_min NaN). They follow from ps-car: car-links (data-upstream, OSRM by "
             "car-times.py) has no row where no road joins the two FUAs. FUAs by the "
             "number of destinations they reach by car (ps-car rows out of 383):", ""]
    rows = []
    for k, ids in groups.items():
        names = ", ".join(name_of(x, i) for i in sorted(ids)) if len(ids) <= 12 else \
            f"{len(ids)} FUAs (" + ", ".join(name_of(x, i) for i in sorted(ids)[:6]) + " ...)"
        rows.append((int(k), len(ids), names))
    lines += [md(pd.DataFrame(rows, columns=["car rows out", "FUAs", "which"])), ""]
    ex = ctx["car_examples"]
    fi = ctx.get("car_fi", {})
    lines += [
        "Reading. " + car_islands_text(x, ctx) + " The larger groups reach the mainland "
        "only through the three crossings car-times.py adds (the Channel, the Strait of "
        "Messina, the Gulf of Finland; network.CAR_CROSSINGS) and have no row to a FUA "
        "that needs a second crossing"
        + (f" (e.g. {ex})" if ex else "") + ". The car stays out of these pairs; their "
        "alternatives are service paths only.", ""]
    if fi:
        lines += [
            f"For Finland this is a gap in the road data, not geography: all {fi['n']} car "
            f"rows from Finland to Sweden or Norway are marked fixed (ps-car), that is they "
            "go through the Gulf of Finland crossing and round the Baltic Sea, "
            f"{fi['min'] / 60:,.0f} to {fi['max'] / 60:,.0f} hours (median "
            f"{fi['median'] / 60:,.0f}); Oulu -> Stockholm takes {fi['oulu_min']:,.0f} min "
            f"over {fi['oulu_km']:,.0f} km. The road network has no land border at "
            f"Tornio-Haparanda (item {item_no('car')}).", ""]
    return "\n".join(lines)


def car_islands_text(x: dict, ctx: dict, small: int = 10) -> str:
    """The FUAs with at most ``small`` car rows out, grouped by the FUAs they
    reach (ps-car), in words."""
    cov = ctx["car_cov"]
    car = x["car"]
    few = cov.loc[cov["out"] <= small, "fua"].astype(int).tolist()
    icc = x["fuas"].set_index("id")["icc"]

    def names(ids):
        w = [f"{name_of(x, i)} ({icc.get(i, '?')})" for i in ids]
        return w[0] if len(w) == 1 else ", ".join(w[:-1]) + " and " + w[-1]
    none = sorted((f for f in few if not (car["o"] == f).any()), key=lambda i: name_of(x, i))
    parts = [f"{names(none)} reach{'es' if len(none) == 1 else ''} no FUA by car"] if none else []
    seen = set(none)
    for f in sorted(few, key=lambda i: name_of(x, i)):
        if f in seen:
            continue
        reach = sorted(int(v) for v in car.loc[car["o"] == f, "d"])
        group = sorted({f, *reach} & set(few))
        seen |= set(group)
        parts.append(f"{names(group)} reach{'es' if len(group) == 1 else ''} "
                     + ("only each other" if set(reach) <= set(group) else
                        "only " + ", ".join(name_of(x, i) for i in reach)))
    return "; ".join(parts) + "." if parts else ""


def section_basis(x: dict, ctx: dict) -> str:
    alts = x["alts"]
    cp = ctx["common"]
    tab = basis_table(alts, cp["disjoint"].to_numpy())
    is_T = alts["basis"].astype(str) == "T"
    n_T, n_dis = int(is_T.sum()), int((is_T & cp["disjoint"]).sum())
    ground = alts["cls"].isin(["B", "R", "BR"])
    n_gT = int((is_T & ground).sum())
    n_gdis = int((is_T & ground & cp["disjoint"]).sum())
    ex_rows = ctx["basis_examples"]
    lines = ["## 8. Basis of the path frequency", "",
             "A path's frequency and existence are counted date by date (basis D) when "
             "its links share at least MIN_COMMON_DATES = "
             f"{x['params'].get('MIN_COMMON_DATES')} active dates, otherwise on typical "
             "weekdays (basis T: per weekday the departures that run on at least "
             f"TYPICAL_SHARE = {x['params'].get('TYPICAL_SHARE')} of the link's active "
             "dates of that weekday, within one IATA season); always on typical days "
             "when a transfer joins air and ground (pathset/transfer.py, 'Basis'). One-link "
             "paths are basis D. Per mode class of the kept alternatives "
             "(ps-alternatives basis; T_disjoint: basis T and the links' [first_day, "
             "last_day] ranges in ps-links have no date in common):", "",
             md(tab.rename(columns={"cls": "class"})), "",
             f"{n_(n_T)} kept alternatives are on typical days; for {n_(n_dis)} of them "
             f"({pct(n_dis, n_T)}) the links never ran in a common period: the chain "
             "joins a flight of one period with a train or coach of another, or two ground "
             "services of different periods. Such a path exists on no real date. This "
             "follows decision 3 (all dates, no common date needed; plan, 'Your "
             "decisions') and the typical-day fallback.", "",
             f"Air with coach: of the {n_(ctx['air_coach']['n'])} kept alternatives on "
             "typical days that use a flight and a coach (classes AB and ABR), "
             f"{n_(ctx['air_coach']['disjoint'])} ({pct(ctx['air_coach']['disjoint'], ctx['air_coach']['n'])}) "
             f"have links with no common period; {n_(ctx['air_coach']['n'] - ctx['air_coach']['disjoint'])} "
             "do, through the coach links that started before 2026. It is not only air "
             f"with ground: {n_(n_gdis)} of the {n_(n_gT)} ground-only (B, R, BR) "
             "alternatives on typical days are also disjoint, because two ground links "
             f"that share fewer than MIN_COMMON_DATES = {x['params'].get('MIN_COMMON_DATES')} "
             "dates fall back to typical days too. No basis D alternative is disjoint "
             f"({int((~is_T & cp['disjoint'] & (alts['seq'].astype(str) != 'C')).sum())}).",
             "", "Three examples (kept alternatives of well-known pairs where possible; "
             "dates are ps-links first_day and last_day of each link):", ""]
    lines += [f"- {r}" for r in ex_rows] + [""]
    return "\n".join(lines)


def pair_line(x: dict, o: int, d: int, alts: pd.DataFrame, best: pd.DataFrame,
              suspect: dict | None = None) -> str:
    """One plain line on a pair from the data: car, direct modes in R0 and
    kept (and how a direct mode with R0 candidates left the set,
    ``direct_fate``), the fastest kept service alternative (and the fastest
    without a link of ``suspect``)."""
    seqs = set(alts["seq"].astype(str))
    r0 = set(best["seq"].astype(str))
    word = {"A": "air", "B": "coach", "R": "rail"}
    parts = []
    for m in ("R", "A", "B"):
        if m in seqs:
            parts.append(f"direct {word[m]} kept")
        elif m in r0:
            parts.append(f"direct {word[m]} in R0 but not kept: "
                         + direct_fate(m, alts, best, x["params"]))
        else:
            parts.append(f"no direct {word[m]} in the data")
    svc = alts[alts["seq"].astype(str) != "C"]
    car = alts[alts["seq"].astype(str) == "C"]
    line = "; ".join(parts) + ". "
    bad = uses_links(svc, suspect) if suspect else np.zeros(len(svc), dtype=bool)
    if len(svc):
        r = svc.sort_values("T").iloc[0]
        line += (f"Fastest kept service alternative {r['seq']} {hm(r['T'])} at "
                 f"{float(r['freq']):g} a day ({r['basis']})")
        if bad[svc.index.get_indexer([r.name])[0]]:
            good = svc[~bad].sort_values("T")
            line += (", on a faulty link (marked !)"
                     + (f"; without faulty links {good['seq'].iloc[0]} "
                        f"{hm(good['T'].iloc[0])} at {float(good['freq'].iloc[0]):g} a day"
                        if len(good) else ""))
            r = good.iloc[0] if len(good) else r
        line += ". "
        slow = int((svc["T"] >= 2 * float(r["T"])).sum())
        if slow:
            line += (f"{slow} of {len(svc)} service alternatives "
                     f"{'takes' if slow == 1 else 'take'} at least twice that time. ")
    line += (f"Car {hm(car['T'].iloc[0])}." if len(car) else "No car alternative.")
    return line


def hole_note(a: str, b: str) -> str:
    """The data holes known for a well-known pair (by the names of
    WELL_KNOWN_PAIRS): the plan's list and the review's."""
    out = []
    if (a, b) in PLAN_KNOWN_HOLES:
        out.append(f"Known data hole (plan, Verification 2): {PLAN_KNOWN_HOLES[(a, b)]}.")
    if (a, b) in REVIEW_HOLES:
        text, item = REVIEW_HOLES[(a, b)]
        out.append(f"Data hole not in the plan's list: {text} (item {item_no(item)}).")
    return " ".join(out)


def section_pairs(x: dict, ctx: dict) -> str:
    alts, best = x["alts"], x["best"]
    p = x["params"]
    suspect = ctx.get("suspect", {})
    lines = ["## 9. Well-known pairs", "",
             "Kept alternatives (ps-alternatives) of each pair: rank, mode sequence, T "
             "(clock time door to door, h:mm), gtt (weighted minutes), gtc (gtt plus the "
             f"hidden wait BETA_HWT / (2 x freq) = {p.get('BETA_HWT')} / (2 x freq), at most "
             f"HWT_MAX = {p.get('HWT_MAX')} min), freq/day (path departures per day: the "
             "median, over the dates or typical days on which the chain runs, of its "
             "departures that day, at most the largest daily count of its sparsest link; "
             "cost.capped_frequency), pooled/day (freq_pooled: departures pooled over the "
             "distinct first links of the alternative's members, rules.pooled_freq), basis "
             "(D dates, T typical days), route (hub names; ' > ' a ride, ' ~ ' a move "
             "between two hubs of a transfer; pathset.view.route).", "",
             "The rank follows gtc, not T or gtt, so a slow frequent path can rank above "
             "a fast rare one. '!' after T marks a path on a faulty link (items "
             f"{item_no('eurostar')} and {item_no('fast_links')}). The line under each "
             "table is read off the data: direct modes from ps-best-by-sequence (R0, "
             "before R3) and ps-alternatives; the known data holes of the pair follow "
             f"(plan, Verification 2, and item {item_no('holes')}).", ""]
    notes = []
    for o, d, uo, ud, a, b in ctx["pairs"]:
        if o is None or d is None:
            lines += [f"### {a} -> {b}", "", "Name not resolved; pair left out.", ""]
            continue
        used = [f"'{u}' for {n}" for u, n in ((uo, a), (ud, b)) if u != n]
        if used:
            notes.append(f"{a} -> {b}: " + ", ".join(used))
        al = alts[(alts["o"] == o) & (alts["d"] == d)].sort_values("rank")
        bs = best[(best["o"] == o) & (best["d"] == d)]
        al = with_routes(al, x)
        bad = uses_links(al, suspect) if suspect else np.zeros(len(al), dtype=bool)
        tab = pd.DataFrame({"rank": al["rank"].astype(int).to_numpy(),
                            "seq": al["seq"].to_numpy(),
                            "T": [hm(t) + (" !" if f else "") for t, f in zip(al["T"], bad)],
                            "gtt": al["gtt"].round(0).astype(int).to_numpy(),
                            "gtc": al["gtc"].round(0).astype(int).to_numpy(),
                            "freq/day": [("" if not np.isfinite(v) else f"{v:g}")
                                         for v in al["freq"].astype(float)],
                            "pooled/day": [("" if not np.isfinite(v) else f"{v:g}")
                                           for v in al["freq_pooled"].astype(float)],
                            "basis": al["basis"].astype(str).to_numpy(),
                            "route": al["route"].to_numpy()})
        hole = hole_note(a, b)
        lines += [f"### {name_of(x, o)} -> {name_of(x, d)} ({o} -> {d})", "",
                  md(tab), "", pair_line(x, o, d, al, bs, suspect)
                  + (" " + hole if hole else ""), ""]
    if notes:
        lines.insert(2, "Names resolved with the local form (pathset.view.resolve_fua "
                     "folds accents, not transliterations): " + "; ".join(notes) + ".\n")
    return "\n".join(lines)


def section_sensitivity(x: dict, ctx: dict) -> str:
    sens, runs = x["sens"], x["sens_runs"]
    lines = ["## 10. Sensitivity on Wien and Katowice", ""]
    method = [
        "Method. params = the run's (pathset-params.yml); the network is built once "
        "(search.load_network). For each origin one search at MU = "
        f"{SENS_SEARCH_MU} (search.candidates towards every other FUA). Per destination "
        "the candidates are numbered as search.decide numbers them (view.number), the "
        "link frequencies are added (view.link_freqs with ps-links), and every variant "
        "runs on the same candidates (view.decide_variant; stagecd.sensitivity_pair).", "",
        "Why one search serves MU 0.25, 0.5 and 1.0: R3 in rules.apply_rules cuts "
        "against the least gtt and the least gtc of each mode class among the "
        "candidates given. The candidates that hold those minima are never cut by the "
        "search, so they are in the MU = 1.0 set, and R3 at a smaller MU on that set "
        "equals R3 on all R0 candidates (checked in pathset/rules.py and "
        "pathset/search.py; module docstring of pathset/stagecd.py).", "",
        "The frequency variants change gtc, so for them this is only guaranteed where "
        "every class satisfies stagecd.r3_exact. Column 'not exact' counts the pairs "
        "where it is not guaranteed; there the variant may miss a candidate that a "
        "search at MU = inf would show.", "",
        "Left out:", ""] + [f"- {k}: {v}." for k, v in SENS_LEFT_OUT.items()]
    lines += method + [""]
    if sens is None or runs is None:
        lines += ["Not run: the sensitivity tables are missing (stage-cd/sensitivity, "
                  "stage-cd/sensitivity-runs). Run `python -X utf8 pathset-stagecd.py "
                  "--sensitivity` (about ten minutes for the two origins on this "
                  "machine, one search process).", ""]
        return "\n".join(lines)
    ok = all(str(v) == "pass" for v in runs["default_check"]) and \
        all(str(v) == "pass" for v in runs["r3_check"])
    lines += [("Checks: " + "; ".join(
        f"{name_of(x, r.origin)}: {n_(r.candidates)} candidates at MU = {r.search_mu:.1f} in "
        f"{r.search_s:,.0f} s; R3 of the default variant keeps per pair exactly ps-funnel "
        f"n_r3: {r.r3_check}; default variant against the stored ps-alternatives "
        f"(same d, rank, key, seq, T, gtt): {r.default_check} "
        f"({n_(r.default_alternatives)} and {n_(r.stored_alternatives)} rows)"
        for r in runs.itertuples()) + "."
        + ("" if ok else " THE DEFAULT CHECK FAILED: the numbers below are not those of "
           "the run; see stage-cd/sensitivity-default-mismatch.")), ""]
    lines += [sensitivity_provenance(x, ctx), ""]
    for o in sens["origin"].unique():
        s = sens[sens["origin"] == o].copy()
        base = s[s["variant"] == "default"].iloc[0]
        tab = pd.DataFrame({
            "variant": s["variant"].to_numpy(),
            "alternatives": s["n_alt"].astype(int).to_numpy(),
            "per pair": s["per_pair"].to_numpy(),
            "pairs capped": s["pairs_capped"].astype(int).to_numpy(),
            "capped": s["n_capped"].astype(int).to_numpy(),
            "before cap": s["before_cap"].astype(int).to_numpy(),
            "default kept (d, key)": [pct(v, 1, 1) if np.isfinite(v) else "" for v in s["share_same_key"]],
            "default found as member": [pct(v, 1, 1) if np.isfinite(v) else "" for v in s["share_found_member"]],
            "not exact": s["pairs_not_exact"].astype(int).to_numpy(),
            "seconds": s["seconds"].to_numpy()})
        cls = s[["variant"] + [f"cls_{k}" for k in CLASSES]].rename(
            columns={f"cls_{k}": k for k in CLASSES})
        lines += [f"### {name_of(x, o)} ({o}): {int(base['pairs'])} destinations", "",
                  md(tab, 2), "", "Alternatives by mode class:", "", md(cls), ""]
    lines += ["What each variant changes:", ""]
    lines += [f"- {k}: {v}" for k, v in SENS_NOTES.items()] + [""]
    lines += ["default kept (d, key): share of the default alternatives whose "
              "representative (destination and link ids) is also a representative under "
              "the variant; default found as member: share whose representative is a "
              "member of some alternative of the variant (it may represent another group "
              "there). seconds: the rules only, summed over the destinations.", ""]
    lines += ["Lessons:", ""] + [f"- {t}" for t in sensitivity_lessons(x, ctx)] + [""]
    return "\n".join(lines)


def sensitivity_provenance(x: dict, ctx: dict) -> str:
    """Which parameters, inputs and code made the saved sensitivity tables,
    against the present ones (columns params_hash, code_hash, written,
    rss_mb and peak_mb of sensitivity-runs; absent in tables written before
    they were added)."""
    runs = x["sens_runs"]
    shard = sorted({a.get("param_hash") for a in x["attrs"].values()})
    code_now = sens_code_hash()
    if "code_hash" not in runs.columns:
        return ("Provenance: these tables carry no hash of their parameters or code (they "
                "were written before the hashes were stored); rerun --sensitivity to tie "
                "them to the code.")
    ph = sorted(set(runs["params_hash"].astype(str)))
    ch = sorted(set(runs["code_hash"].astype(str)))
    mem = "; ".join(f"{name_of(x, r.origin)} {r.rss_mb:,.0f} MB resident after the origin, "
                    f"peak {r.peak_mb:,.0f} MB" for r in runs.itertuples()
                    if hasattr(r, "rss_mb"))
    return (f"Provenance: written {runs['written'].iloc[0]} (sensitivity-runs). Parameters and "
            f"inputs: run hash {', '.join(ph)}, "
            + ("equal to the stage D shards'" if ph == shard else
               f"NOT the stage D shards' ({', '.join(str(h) for h in shard)})")
            + f". Code: {', '.join(ch)}, "
            + ("equal to the present code's (stagecd.sens_code_hash)" if ch == [code_now] else
               f"NOT the present code's ({code_now}): rerun --sensitivity")
            + (f". Memory of the one search process: {mem}." if mem else "."))


def no_r4_mix(sens_alts: pd.DataFrame, origin: int) -> dict:
    """What cross_class_no_r4 keeps for one origin (sensitivity-alts): all,
    car, direct flights (seq A), two flights (seq AA), others, and the share
    of car and direct flights."""
    a = sens_alts[(sens_alts["origin"] == origin)
                  & (sens_alts["variant"] == "cross_class_no_r4")]
    vc = a["seq"].astype(str).value_counts()
    n, car, one, two = len(a), int(vc.get("C", 0)), int(vc.get("A", 0)), int(vc.get("AA", 0))
    return {"n": n, "car": car, "A": one, "AA": two, "other": n - car - one - two,
            "share": (car + one) / n if n else math.nan}


def sensitivity_lessons(x: dict, ctx: dict) -> list:
    """Plain sentences read off the sensitivity tables; orders, shares and
    counts are computed from them."""
    s = x["sens"]
    sa = x.get("sens_alts")
    runs = x["sens_runs"]
    names = list(s["origin"].unique())
    per = ctx["link_periods"]

    def val(o, v, col):
        return s[(s["origin"] == o) & (s["variant"] == v)][col].iloc[0]

    def both(v, col="share_same_key", fmt=None):
        fmt = fmt or (lambda z: pct(z, 1, 1))
        return ", ".join(f"{name_of(x, o)} {fmt(val(o, v, col))}" for o in names)

    out = []
    for o in names:
        out.append(
            f"{name_of(x, o)}: the cap acts in {int(val(o, 'default', 'pairs_capped'))} of "
            f"{int(val(o, 'default', 'pairs'))} pairs and drops "
            f"{int(val(o, 'default', 'n_capped')):,} alternatives; without it (no_cap) the "
            f"sets would hold {int(val(o, 'no_cap', 'n_alt')):,} alternatives "
            f"({val(o, 'no_cap', 'per_pair'):.1f} a pair) instead of "
            f"{int(val(o, 'default', 'n_alt')):,} ({val(o, 'default', 'per_pair'):.1f}).")
    out.append("In a capped pair the set size is the cap whatever the rules do, so there a "
               "variant can change only which alternatives are kept, not how many.")
    for o in names:
        r = rank_variants(s, o)
        out.append(f"{name_of(x, o)}, the variants from the largest change of the set to the "
                   "smallest (share of the default alternatives kept as such): "
                   + ", ".join(f"{v} {pct(z, 1, 1)}" for v, z in r) + ".")
    out.append(
        f"The time band: the flat 15-minute band (eps_rel_0, no 5% term) keeps "
        f"{both('eps_rel_0')} of the default alternatives as such, so the 5% term decides "
        f"about a tenth of the set; the absolute part matters little (eps_10 keeps "
        f"{both('eps_10')}, eps_20 {both('eps_20')}).")
    out.append(oded_lesson(x, ctx))
    out.append(
        f"The margin: mu_0.25 keeps {both('mu_0.25')} (alternatives "
        f"{both('mu_0.25', 'n_alt', n_)}), mu_1.0 {both('mu_1.0')} (alternatives "
        f"{both('mu_1.0', 'n_alt', n_)}), against default {both('default', 'n_alt', n_)}.")
    out.append(
        f"The gate: rho_0.33 keeps {both('rho_0.33')}, rho_1.0 {both('rho_1.0')}; gate_path "
        f"{both('gate_path')}, gate_pooled {both('gate_pooled')}. Under gate_pooled the cap "
        f"acts in {both('gate_pooled', 'pairs_capped', n_)} pairs (default: "
        f"{both('default', 'pairs_capped', n_)}) and drops "
        f"{both('gate_pooled', 'n_capped', n_)} alternatives: with the plan's pooled gate "
        "the cap, not R2, would choose most sets.")
    out.append(
        f"Keying R1 differently: r1_class keeps {both('r1_class')} as such (found as a "
        f"member: {both('r1_class', 'share_found_member')}), r1_pattern {both('r1_pattern')}; "
        f"R2 on weighted time (r2_weighted) {both('r2_weighted')}.")
    mix = ({o: no_r4_mix(sa, o) for o in names} if sa is not None else {})
    out.append(
        f"Comparing across classes removes whole classes: cross_class leaves "
        f"{both('cross_class', 'n_alt', n_)} alternatives (R4 gives each class its best "
        f"survivor back), cross_class_no_r4 {both('cross_class_no_r4', 'n_alt', n_)}. "
        + ("Without R4 what is left is mainly the car and the direct flights (sensitivity-"
           "alts, by mode sequence): " + "; ".join(
               f"{name_of(x, o)} {m['car']:,} car, {m['A']:,} direct flights, {m['AA']:,} "
               f"with two flights (AA), {m['other']:,} others (car and direct flights "
               f"{pct(m['share'], 1, 1)})" for o, m in mix.items()) + ". " if mix else "")
        + "With zero bands an alternative of another class removes every one with at least "
        "as many transfers that is not faster (one of the two strictly), when it passes "
        "the gate, and the car (no transfer, infinite frequency) passes every gate.")
    first_two = []
    for o in names:
        r = [v for v, _ in rank_variants(s, o, skip=("default", "no_cap", "cross_class_no_r4",
                                                      "freq_fixed"))][:2]
        first_two.append(set(r))
    lead = (all(t == {"freq_first", "freq_span"} for t in first_two))
    out.append(
        ("After cross_class_no_r4, freq_first and freq_span change the set most in both "
         "origins (freq_fixed left aside): " if lead else "The frequency definitions: ")
        + f"freq_first keeps {both('freq_first')}, freq_span {both('freq_span')}. freq_fixed "
        f"keeps {both('freq_fixed')}, but it mostly measures the feed dates: ps-links "
        f"freq_fixed (2023-2025 departures over the days of 2023-2025) is 0 for "
        f"{n_(per['B']['fixed_zero'])} of {n_(per['B']['n'])} coach links and "
        f"{n_(per['R']['fixed_zero'])} of {n_(per['R']['n'])} rail links, those whose data "
        f"start in 2026 ({n_(per['A']['fixed_zero'])} air links). A path on such a link "
        f"gets the largest hidden wait (HWT_MAX), can remove nothing through the gate, and "
        "any noticeably better path of its class removes it (0 >= RHO_F x 0). These three "
        "are the least certain numbers here: equality with a full search is not "
        f"guaranteed in {both('freq_fixed', 'pairs_not_exact', n_)} pairs (freq_fixed), "
        f"{both('freq_span', 'pairs_not_exact', n_)} (freq_span) and "
        f"{both('freq_first', 'pairs_not_exact', n_)} (freq_first).")
    return out


def oded_lesson(x: dict, ctx: dict, short: bool = False) -> str:
    """Oded's shape on the sensitivity origins: R2 removals of the default by
    a remover with fewer transfers that is later by more than EPS_T_ABS, as
    a share of all R2 removals (sensitivity-runs; checked against ps-funnel
    n_r1 - n_r2), with the bands of those removals; ``short``: the counts
    only."""
    runs, f = x["sens_runs"], x["funnel"]
    eps_abs = float(x["params"]["EPS_T_ABS"])
    parts, equal = [], True
    for r in runs.itertuples():
        fo = f[f["o"] == r.origin]
        stored = int((fo["n_r1"] - fo["n_r2"]).sum())
        tot = int(getattr(r, "r2_removals", stored))
        equal &= tot == stored
        txt = (f"{name_of(x, r.origin)} {int(r.oded_removals):,} of {tot:,} R2 removals "
               f"({pct(r.oded_removals, tot, 2)})")
        if short:
            parts.append(txt)
            continue
        if hasattr(r, "oded_eps_max"):
            txt += (f", bands up to {float(r.oded_eps_max):.1f} min (all R2 removals up to "
                    f"{float(r.r2_eps_max):.1f})")
        if tot != stored:
            txt += f"; ps-funnel gives {stored:,} removals, NOT the same"
        parts.append(txt)
    if short:
        return "Oded's shape is rare: " + "; ".join(parts) + "."
    kept_band = ", ".join(f"{name_of(x, r.origin)} {float(r.eps_max):.1f}"
                          for r in runs.itertuples())
    if "oded_eps_max" in runs.columns:
        tail = (" The largest of these bands is close to the widest band of any class with a "
                f"kept service alternative ({kept_band} min; 0.05 x a best T near 24 h). How "
                "much later each remover was (at most its band) was not stored.")
    else:
        tail = (f" The widest band of a class with a kept service alternative is {kept_band} "
                "min; the bands of the removals themselves were not stored by this "
                "sensitivity run.")
    return ("Oded's shape (a remover with fewer transfers that is later by more than "
            f"EPS_T_ABS = {eps_abs:g} min, so only the 5% term let it remove) is rare: "
            + "; ".join(parts)
            + ((". sensitivity-runs r2_removals equal ps-funnel n_r1 - n_r2 of each origin."
                if equal else ".") if "r2_removals" in runs.columns else
               " (removals from ps-funnel n_r1 - n_r2).")
            + tail)


def section_open(x: dict, ctx: dict) -> str:
    c = ctx["open"]
    p = x["params"]
    lines = ["## 11. Decisions to confirm and open items", "",
             "Each item: what was seen, what I checked now (with the source), what is "
             "open.", "", "Overnight decisions to confirm:", ""]
    lines += [f"- PASSED_SHARE {p.get('PASSED_SHARE')}. A link 'passes' a city when the runs "
              "calling there carry at least half of its dated departures; R0 rejects a path "
              "whose link passes o, d or a city of the path. Checked: " + c["pass_through"]
              + " Open: whether a path whose train calls at the destination on some runs "
              "should stay (it does now, as the runs are not split by stop pattern)."]
    lines += [f"- R2_GATE {p.get('R2_GATE')}. The run's gate: p may remove q only if freq(p) "
              f">= RHO_F x the largest path frequency among the members of q's R1 group "
              "(section 1). The plan's wording is 'pooled' (the pooled frequency of q's "
              "group); 'path' uses q's representative. Checked: " + c["gate"]
              + " Open: top, path or pooled."]
    lines += [f"- WORKERS {p.get('WORKERS')} (the plan said six). Checked: " + c["workers"]
              + " Open: whether to use five or six workers next time."]
    lines += ["", "Open items found during the night:", ""]
    if len(c["items"]) != N_NIGHT_ITEMS:
        raise RuntimeError(f"expected {N_NIGHT_ITEMS} night items, got {len(c['items'])}")
    lines += [f"{i}. {t}" for i, t in enumerate(c["items"], 1)]
    lines += ["", "Found by the review of the first version of this report (14 Nov), in the "
              "data and the design. Each says what was seen and what is open; the numbers "
              "are recomputed here.", ""]
    lines += [f"{item_no(k)}. {c['review'][k]}" for k in REVIEW_ITEMS]
    lines += ["", "Possible problems in the enumeration:", "",
              "- None from the review: no finding was about the code of the search or the "
              f"rules (search, rules, pairs, cost). Item {item_no('eurostar')} and the Lyon "
              f"case of item {item_no('fast_links')} are errors in timetable feeds that the "
              "network build passes on (a long dwell at a call becomes a short ride); the "
              f"other fast links of item {item_no('fast_links')} were not traced to a cause. "
              f"Items {item_no('near_ties')}, {item_no('distance')} and {item_no('car')} are "
              "questions about the rules and their scope, not bugs."]
    lines += ["", "New in this report:", ""] + [f"- {t}" for t in c["new"]] + [""]
    return "\n".join(lines)


# --- Report ------------------------------------------------------------------
def machine_text() -> str:
    """RAM, logical CPUs and the operating system of this machine."""
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2 ** 30:.1f} GB RAM"
        cpu = f"{psutil.cpu_count(logical=True)} logical CPUs"
    except ImportError:
        ram, cpu = "RAM unknown", "CPUs unknown"
    return (f"{ram}, {cpu}, {platform.system()} {platform.release()} "
            f"(version {platform.version()}); the run and this report on the same machine")


def car_missing_example(x: dict) -> str:
    """A pair that needs two crossings and has no car row (London ->
    Helsinki), when it is indeed missing."""
    car = x["car"]
    try:
        lo, he = V.resolve_fua("London", x["fuas"]), V.resolve_fua("Helsinki", x["fuas"])
    except ValueError:
        return ""
    has = ((car["o"] == lo) & (car["d"] == he)).any()
    return "" if has else f"London -> Helsinki has no ps-car row"


def basis_examples(x: dict, ctx: dict) -> list:
    """Three T-basis alternatives whose links share no period: from the
    well-known pairs (lowest rank first, distinct pairs and classes), else
    the largest gaps."""
    alts, cp, links = x["alts"], ctx["common"], x["links"].set_index("link")
    t = alts[(alts["basis"].astype(str) == "T") & cp["disjoint"].to_numpy()]
    wk = {(o, d) for o, d, *_ in ctx["pairs"] if o is not None and d is not None}
    cand = t[[(int(o), int(d)) in wk for o, d in zip(t["o"], t["d"])]].sort_values(["rank", "o", "d"])
    pick, seen_p, seen_c = [], set(), set()
    for i, r in cand.iterrows():
        if (r["o"], r["d"]) in seen_p or r["cls"] in seen_c:
            continue
        pick.append(i)
        seen_p.add((r["o"], r["d"]))
        seen_c.add(r["cls"])
        if len(pick) == 3:
            break
    if len(pick) < 3:
        gap = (cp.loc[t.index, "first"] - cp.loc[t.index, "last"]).sort_values(ascending=False)
        pick += [i for i in gap.index if i not in pick][:3 - len(pick)]
    rows = with_routes(alts.loc[pick], x)
    out = []
    for _, r in rows.iterrows():
        dates = "; ".join(
            f"link {int(l)} ({links.loc[int(l), 'mode']}) {D.index_date(links.loc[int(l), 'first_day'])}"
            f" to {D.index_date(links.loc[int(l), 'last_day'])}"
            for l in (r["l1"], r["l2"], r["l3"]) if l >= 0)
        out.append(f"{name_of(x, r['o'])} -> {name_of(x, r['d'])}, rank {int(r['rank'])}, "
                   f"{r['seq']} '{r['route']}', T {hm(r['T'])}, freq {float(r['freq']):g} a "
                   f"day on typical days: {dates}.")
    return out


def open_items(x: dict, ctx: dict) -> dict:
    """The checks of section 11 (cheap ones only)."""
    p, alts, links = x["params"], x["alts"], x["links"]
    fuas = x["fuas"]
    out = {}
    # pass-through (decision PASSED_SHARE; item 8)
    svc = alts[alts["seq"].astype(str) != "C"]
    lk = links.sort_values("link")
    hubs_cols = V.link_hubs(svc[["o", "d", "l1", "l2", "l3"]], lk["src"].to_numpy(),
                            lk["trg"].to_numpy())
    ptr0, pc0 = load_passed(x, 0.0)
    any_run = passes_through(hubs_cols, x["city"], ptr0, pc0)
    ptr5, pc5 = load_passed(x, float(p["PASSED_SHARE"]))
    at_share = passes_through(hubs_cols, x["city"], ptr5, pc5)
    named = {(int(a), int(b)) for a, b in p["NAMED_PAIRS"]}
    is_named = np.array([(int(o), int(d)) in named for o, d in zip(svc["o"], svc["d"])])
    o7 = svc["o"].isin([7, 133]).to_numpy()
    out["pass_through"] = (
        f"with the passed cities rebuilt from ps-events at share 0 (any run with a date; "
        f"search.passed_cities), {n_(any_run.sum())} of {n_(len(svc))} kept service "
        f"alternatives ({pct(any_run.sum(), len(svc))}) have a link that passes o, d or "
        f"a city of the path on some of its runs, but on fewer than half of its dated "
        f"departures. For origins 7 and 133 (Amsterdam, Helsinki), all destinations: "
        f"{n_(any_run[o7].sum())} of {n_(o7.sum())}. The night's {NIGHT_PASS_THROUGH_NAMED} "
        "probably came from the same test on those two origins after their run of 22:05 "
        "(R2_GATE path; the check script dates from 22:11), so 'named pairs' in the item "
        "would mean their origins. On the ten named pairs "
        f"themselves: {n_(any_run[is_named].sum())} of {n_(is_named.sum())}. The same test "
        f"at share {p['PASSED_SHARE']} finds {n_(at_share.sum())}, as it must (the search "
        "applied it; this checks the test).")
    ctx["pass_any"] = int(any_run.sum())
    out["pass_short"] = (
        f"{n_(any_run[o7].sum())} of {n_(o7.sum())} kept service alternatives of origins 7 "
        f"and 133 (probably the night's scope; it counted {NIGHT_PASS_THROUGH_NAMED} on an "
        f"earlier run), {n_(any_run.sum())} of {n_(len(svc))} over all pairs, "
        f"{n_(any_run[is_named].sum())} on the ten named pairs (details under the decision "
        "PASSED_SHARE above).")
    # gate
    sb = x["stage_b"]
    gate = []
    if sb is not None:
        g = sb.groupby("variant")[["n_alt", "n_capped"]].sum()
        gate.append(f"stage B (stage-b/variants.parquet, 10 named pairs): capped "
                    f"{int(g.loc['default', 'n_capped'])} under top, "
                    f"{int(g.loc['gate_path', 'n_capped'])} under path, "
                    f"{int(g.loc['gate_pooled', 'n_capped'])} under pooled")
    if x["sens"] is not None:
        s = x["sens"]
        for o in s["origin"].unique():
            t = s[s["origin"] == o].set_index("variant")
            gate.append(f"{name_of(x, o)} (section 10): pairs capped "
                        f"{n_(t.loc['default', 'pairs_capped'])} (top), "
                        f"{n_(t.loc['gate_path', 'pairs_capped'])} (path), "
                        f"{n_(t.loc['gate_pooled', 'pairs_capped'])} (pooled) of "
                        f"{n_(t.loc['default', 'pairs'])}; alternatives capped "
                        f"{n_(t.loc['default', 'n_capped'])}, "
                        f"{n_(t.loc['gate_path', 'n_capped'])}, "
                        f"{n_(t.loc['gate_pooled', 'n_capped'])}")
    out["gate"] = "; ".join(gate) + "." if gate else "no stage B or sensitivity table."
    tt = ctx["timing"]
    mx = tt.loc[tt["peak_mb"].idxmax()]
    ram = None
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2 ** 20
    except ImportError:
        pass
    w6 = worker_estimate(tt["peak_mb"], 6)
    w4 = worker_estimate(tt["peak_mb"], int(x["run"]["info"]["workers"]))
    out["workers"] = (
        f"the largest worker peak was {mx['peak_mb']:,.0f} MB (after {name_of(x, mx['origin'])}, "
        f"{int(mx['origin'])}; {w6['n_top']} origins of that worker show it, as peak_mb only "
        f"grows within a worker); every other origin peaked at {w6['second']:,.0f} MB or "
        f"less. With one worker at the largest peak and the others at the next, four "
        f"workers need about {w4['estimate'] / 1024:.1f} GB and six about "
        f"{w6['estimate'] / 1024:.1f} GB, plus the main process; the worst case, six at "
        f"{w6['top']:,.0f} MB, is {w6['worst'] / 1024:.1f} GB"
        + (f" of {ram / 1024:.1f} GB" if ram else "") + ". The run took "
        f"{wall_text(ctx['wall'])} with {x['run']['info']['workers']} workers, at "
        f"{ctx['use']:.0%} use of them (section 3).")
    items = [safe_item(f, x, ctx, out, label) for f, label in (
        (item_amsterdam_berlin, "Amsterdam-Berlin"), (item_helsinki_car, "Helsinki-Tallinn by car"),
        (item_typical_disjoint, "Air-ground chains on typical days"),
        (item_detours, "R4 keeps detours"), (item_rail_replacement, "Rail-replacement buses"),
        (item_oded, "Oded's case"), (item_trace, "The trace names only the first remover"),
        (item_pass_through, "Pass-through"), (item_run_hash, "The run hash"),
        (item_ftrans, "pairs.ftrans_table"), (item_earlier, "Earlier choices"))]
    out["items"] = items
    out["review"] = review_items(x, ctx)
    out["new"] = new_findings(x, ctx)
    return out


def safe_item(fn, x: dict, ctx: dict, out: dict, label: str) -> str:
    """The text of one open item; a check that fails gives a line that says
    so, so that the numbering of the items never shifts."""
    try:
        return fn(x, ctx, out)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        return f"{label}. Not checked now: {type(e).__name__}: {e}."


def item_amsterdam_berlin(x: dict, ctx: dict, out: dict) -> str:
    fuas, links, alts = x["fuas"], x["links"], x["alts"]
    a_id, b_id = V.resolve_fua("Amsterdam", fuas), V.resolve_fua("Berlin", fuas)
    hubs = x["hubs"]
    hA, hB = hubs.loc[hubs["fua"] == a_id, "hub"], hubs.loc[hubs["fua"] == b_id, "hub"]
    l = links[links["src"].isin(hA) & links["trg"].isin(hB) & (links["mode"] == "R")]
    l = l.sort_values("ride")
    sp = x["sparse"]
    s = sp[sp["src"].isin(hA) & sp["trg"].isin(hB) & (sp["mode"] == "R")]
    al = alts[(alts["o"] == a_id) & (alts["d"] == b_id)]
    direct = al[al["seq"].astype(str) == "R"]
    rr = al[al["seq"].astype(str).isin(["RR", "RRR"])]
    dd = [D.index_date(v) for v in (l["first_day"].min(), l["last_day"].max())] if len(l) else []
    lk = links.set_index("link")
    txt = ("Amsterdam-Berlin. Seen (night): the Intercity is split at Hengelo in the rail "
           "data (the same trip key in two feeds), so the through train shows as two links; "
           "a through journey exists on only 14 dates. Checked now: ps-links has "
           f"{len(l)} direct rail links from Amsterdam hubs to Berlin hubs: "
           + ", ".join(f"{r:.0f} min on {int(n)} dates" for r, n in zip(l["ride"], l["active_dates"]))
           + (f", between {dd[0]} and {dd[1]}" if dd else "")
           + f"; ps-sparse holds {len(s)} more (fewer than 5 dates). So through trains run on "
           "more than 14 dates; the 14 dates fit only the fastest riding-time class.")
    if len(direct):
        l1 = int(direct["l1"].iloc[0])
        others = sorted({float(v) for v in l.loc[l["link"] != l1, "freq"]})
        txt += (f" The kept direct R rides link {l1}: {int(lk.at[l1, 'active_dates'])} dates, "
                f"{float(direct['freq'].iloc[0]):g} a day on them under the median rule "
                f"(decision 4), where the other links count "
                + " or ".join(f"{v:g}" for v in others) + " a day.")
        tr = x["trace"]
        tp = tr[(tr["o"] == a_id) & (tr["d"] == b_id)]
        rep = tp[(tp["seq"].astype(str) == "R") & (tp["status"].astype(str) == "kept")]
        if len(rep):
            rem = tp[(tp["rule"].astype(str) == "R2") & (tp["by"] == int(rep["cand"].iloc[0]))]
            det = r2_details(rem["detail"])
            faster = det[det["X_p"] > det["X_q"]]
            if len(faster):
                r = faster.sort_values("X_q").iloc[0]
                txt += (" With that frequency it passes the gate and removes, in ps-trace, "
                        + ("a faster candidate" if len(faster) == 1 else
                           f"{len(faster)} faster candidates, e.g. one")
                        + f" through the band: {int(r['n_q'])} transfer"
                        f"{'s' if int(r['n_q']) != 1 else ''}, T {r['X_q']:.0f} against its "
                        f"{r['X_p']:.0f} min (band {r['eps']:.1f} min; Oded's shape, item 6).")
    txt += (f" {len(rr)} kept RR/RRR alternatives carry the rest, e.g. via Zwolle and Hengelo. "
            "The trip keys of the two feeds were not rechecked. Open: whether a link that ran "
            "on 14 dates should count as several a day; a vetted list of through trains "
            "across feed borders (plan, Known limits).")
    return txt


def item_helsinki_car(x: dict, ctx: dict, out: dict) -> str:
    fuas, alts, car = x["fuas"], x["alts"], x["car"]
    h_id, t_id = V.resolve_fua("Helsinki", fuas), V.resolve_fua("Tallinn", fuas)
    r = car[(car["o"] == h_id) & (car["d"] == t_id)].iloc[0]
    r2 = car[(car["o"] == t_id) & (car["d"] == h_id)].iloc[0]
    al = alts[(alts["o"] == h_id) & (alts["d"] == t_id)].sort_values("rank")
    is_car = al["seq"].astype(str) == "C"
    car_rank = int(al.loc[is_car, "rank"].iloc[0]) if is_car.any() else None
    return (
        f"Helsinki-Tallinn by car. Seen: the time is the 87.7-minute ferry crossing "
        f"constant restored from the car-times unit bug. Checked now: ps-car "
        f"Helsinki -> Tallinn {float(r['minutes']):.1f} min, {float(r['km']):.0f} km, "
        f"fixed {bool(r['fixed'])}; back {float(r2['minutes']):.1f} min; "
        "network.CAR_CROSSINGS holds 87.7 and 88.3 min for the crossing. So the car "
        "time is the crossing alone, with no drive to or from the ports, no check-in "
        "and no wait for a sailing; the car is "
        + (f"rank {car_rank} of {len(al)} kept alternatives" if car_rank else "not kept")
        + ". Open: a crossing time with boarding and the drives, and how often a "
        "sailing goes (the car has no hidden wait).")


def item_typical_disjoint(x: dict, ctx: dict, out: dict) -> str:
    fuas, links, alts = x["fuas"], x["links"], x["alts"]
    t_alt = alts[alts["basis"].astype(str) == "T"]
    dis = ctx["common"].loc[t_alt.index, "disjoint"]
    airg = t_alt["cls"].astype(str).isin(["AB", "ABR", "AR"])
    parts = []
    for o, (n_dis, n_t) in NIGHT_T_BASIS.items():
        m = t_alt["o"] == o
        parts.append(f"origin {o}: {n_(dis[m].sum())} of {n_(m.sum())} T-basis alternatives "
                     f"(air-ground and ground-only; noted: {n_dis:,} of {n_t:,}), air-ground "
                     f"only {n_(dis[m & airg].sum())} of {n_((m & airg).sum())}")
    tmp = ""
    try:
        tm, tl = V.resolve_fua("Tampere", fuas), V.resolve_fua("Tallinn", fuas)
        hubs, city = x["hubs"], x["city"]
        air = hubs[hubs["kind"] == "A"]["hub"]
        al = links[(links["mode"] == "A") & links["src"].isin(air) & links["trg"].isin(air)]
        tmp_l = al[(city[al["src"].to_numpy(np.int64)] == tm)
                   & (city[al["trg"].to_numpy(np.int64)] == tl)]
        if len(tmp_l):
            tmp = (f" Tampere -> Tallinn flights: last_day "
                   f"{D.index_date(int(tmp_l['last_day'].max()))} (ps-links, {len(tmp_l)} link).")
    except ValueError:
        pass
    return ("Air-ground chains on typical days join links whose date ranges never overlap "
            "(origin 7: 1,380 of 1,970 T-basis alternatives; origin 133: 1,386 of 1,846; e.g. "
            "Tampere-Tallinn flights ended 2025-03-29). Checked now (section 8, same "
            "definition: the links' [first_day, last_day] have no common date): "
            + "; ".join(parts) + ". The night's script counted all T-basis alternatives, as "
            "the first numbers here do; its figures probably came from an earlier run of "
            "these two origins (the script dates from 21:53, before R2_GATE top and before "
            "stage D)." + tmp
            + " Over all pairs " + n_(int(dis.sum())) + " of " + n_(len(t_alt))
            + " T-basis alternatives. Open: whether such chains should count at all "
            "(decision 3 says no common date is needed).")


def item_detours(x: dict, ctx: dict, out: dict) -> str:
    return ("R4 keeps detours as the last alternative of a class (Amsterdam > Bruxelles ~ "
            "Leuven > coach back; Helsinki > Riga > coach). Checked now: "
            + ctx["detours"]["text"]
            + " Open: a rule against such detours that does not rest on the cap, for "
            "example a margin or a comparison across classes (cross_class in section 10 "
            f"shows what that removes; item {item_no('distance')} shows where they gather).")


def item_rail_replacement(x: dict, ctx: dict, out: dict) -> str:
    return ("Rail-replacement buses (route_type 3 of rail operators) count as coach "
            "(B). Checked now: " + ctx["rail_replacement"] + " Open: whether to move them "
            "to R or drop them.")


def item_oded(x: dict, ctx: dict, out: dict) -> str:
    return (f"Oded's case (one transfer less, 20 minutes later): the 5% term makes eps_T up "
            f"to {NIGHT_EPS_MAX} min on long pairs; stage B had {NIGHT_ODED_STAGE_B} removals "
            "of this shape. Checked now: " + ctx["oded"] + " Open: flat 15 min or the 5% "
            "term (eps_rel_0 in section 10 shows the flat band).")


def item_trace(x: dict, ctx: dict, out: dict) -> str:
    st = remover_status(x["trace"])
    vc = st.value_counts()
    capped = int(vc.get("capped", 0))
    return ("The trace names only the first kept remover of a removed candidate. Checked "
            "now in rules.stable_set: 'by' is the first R2 survivor (in increasing gtc) "
            "that removes the candidate; others that also remove it are not listed. The cap "
            f"comes after R2: of the {n_(len(st))} R2 removals in ps-trace (the named pairs), "
            f"{n_(vc.get('kept', 0))} name a remover that is in the final set and "
            f"{n_(capped)} one that the cap then dropped (ps-trace status of 'by' as 'cand' "
            "of the same pair); for those the named remover is not in the final set, and "
            "whether another kept alternative also removes the candidate is not recorded. "
            "Open: list every remover, or at least every kept one, if you want it.")


def item_pass_through(x: dict, ctx: dict, out: dict) -> str:
    return (f"{NIGHT_PASS_THROUGH_NAMED} kept alternatives on the named pairs still ride "
            "through a city on less than half of their departures (links are not split by "
            "stop pattern). Checked now: " + out["pass_short"]
            + " Open: split links by stop pattern, or accept it (decision PASSED_SHARE).")


def item_run_hash(x: dict, ctx: dict, out: dict) -> str:
    g = ctx.get("git", {})
    untracked = [l[3:] for l in (ctx.get("git_pathset") or "").splitlines()
                 if l.startswith("??")]
    return ("The run hash covers parameters and inputs, not code: shards from older code "
            "look current. Checked now (section 1): the files the search runs (RUN_CODE: "
            + ", ".join(RUN_CODE) + ") were last modified before the run started, and "
            + ("none of them differs from HEAD (git status --porcelain)"
               if g.get("status") == "" else f"git status reports '{g.get('status')}'")
            + (f". pathset/ also holds {', '.join(untracked)}, untracked, which the search "
               "does not import" if untracked else "")
            + ". Open: add a hash of the code to the run hash.")


def item_ftrans(x: dict, ctx: dict, out: dict) -> str:
    calls = ftrans_callers(io.ROOT, FTRANS_SCAN)
    prod = [c for c in calls if "pathset-transfers.py" in c[0]]
    return ("pairs.ftrans_table keeps a days=None fallback for an older test. Checked now: "
            "outside tests/ the only call is "
            + "; ".join(f"{f} line {i}: `{t}`" for f, i, t in prod)
            + " (with days); the fallback serves tests/test_pairs.py and "
            "tests/test_transfers_review.py only. Open: drop the fallback and update those "
            "tests.")


def item_earlier(x: dict, ctx: dict, out: dict) -> str:
    return "Earlier choices to confirm (13 Nov checks). " + ctx["earlier"]


# --- Items 12 to 23: what the review found -------------------------------------
def review_items(x: dict, ctx: dict) -> dict:
    """Item name (REVIEW_ITEMS) -> text: what was seen, the numbers now, what
    is open."""
    funcs = {"eurostar": (rv_eurostar, "Eurostar"),
             "fast_links": (rv_fast_links, "Other links faster than any real service"),
             "holes": (rv_holes, "Thin or missing direct services"),
             "carriers": (rv_carriers, "Air carriers"),
             "class_split": (rv_class_split, "Frequency of one riding-time class"),
             "near_ties": (rv_near_ties, "Near-ties"),
             "distance": (rv_distance, "Set sizes by distance"), "car": (rv_car, "Car"),
             "sea_access": (rv_sea_access, "Fitted access over the sea"),
             "stations": (rv_stations, "Main stations and egress minutes"),
             "tz_coach": (rv_tz_coach, "Coach times across time zones"),
             "rare": (rv_rare, "Rare services read as daily")}
    return {k: safe_item(funcs[k][0], x, ctx, {}, funcs[k][1]) for k in REVIEW_ITEMS}


def _link_text(x: dict, l: int, speed: bool = False) -> str:
    lk = x["links"].set_index("link")
    r = lk.loc[int(l)]
    txt = (f"{x['names'][int(r['src'])]} > {x['names'][int(r['trg'])]} {float(r['ride']):.0f} min "
           f"(link {int(l)}, {int(r['active_dates'])} dates, {float(r['freq']):g} a day")
    if speed:
        txt += f", {float(r['km']):.0f} km, {float(r['km']) / float(r['ride']) * 60:.0f} km/h straight line"
    return txt + ")"


def _usage(x: dict, ids) -> str:
    alts, best = x["alts"], x["best"]
    u = uses_links(alts, ids)
    return (f"{n_(u.sum())} kept alternatives in {n_(alts.loc[u, ['o', 'd']].drop_duplicates().shape[0])} "
            f"pairs ({n_((u & (alts['rank'] == 1).to_numpy()).sum())} at rank 1) and "
            f"{n_(uses_links(best, ids).sum())} ps-best-by-sequence rows (the CASPT input)")


def dwell_suspects(t: pd.DataFrame) -> tuple:
    """(suspect rows, least ride of the others) of a dwell_table: rows with a
    dwell of 15 min or more whose ride from the listed departure is shorter
    than any journey with a dwell under 15 min."""
    short = t[t["dwell"] < 15]
    floor = float(short["ride"].min()) if len(short) else math.inf
    return t[(t["dwell"] >= 15) & (t["ride"] < floor)], floor


def _dwell_text(t: pd.DataFrame | None, where: str) -> str:
    if t is None or not len(t):
        return "the dwell table could not be read (data-upstream-hubs)"
    sus, floor = dwell_suspects(t)
    out = []
    for r in sus.itertuples():
        out.append(f"{n_(r.journeys)} journeys of {r.feed_name} (feed {int(r.feed)}) dwell "
                   f"{int(r.dwell)} min at {where} and ride {int(r.ride)} min from the listed "
                   f"departure, {int(r.ride_from_arr)} from the arrival")
    return ("; ".join(out) + (f"; the journeys with a dwell under 15 min ride at least "
                              f"{floor:.0f} min" if np.isfinite(floor) else ""))


def rv_eurostar(x: dict, ctx: dict, out: dict) -> str:
    sus, links = ctx["suspect"], x["links"]
    ids = [l for l in (5420, 5158) if l in sus]
    if not ids:
        return ("Eurostar. Links 5420 and 5158 are no longer in ps-links as the review saw "
                "them; not checked now.")
    lk = links.set_index("link")
    nxt = []
    for l in ids:
        same = links[(links["src"] == lk.at[l, "src"]) & (links["trg"] == lk.at[l, "trg"])
                     & (links["mode"] == "R")].sort_values("ride")
        nxt.append(float(same["ride"].iloc[1]) if len(same) > 1 else math.nan)
    dw = x["review"]["dwell"]
    return ("Eurostar. Seen: Paris Nord > St-Pancras has a riding-time class of 121 min, "
            "faster than any other run in the data. Checked now: ps-links "
            + "; ".join(_link_text(x, l) for l in ids)
            + "; the next classes take " + " and ".join(f"{v:.0f}" for v in nxt) + " min. In "
            "data-upstream-hubs (ic-timetable, ic-journeys) Paris Nord > St-Pancras: "
            + _dwell_text(dw.get("eurostar"), "Paris Nord") + "; St-Pancras > Paris Nord: "
            + _dwell_text(dw.get("eurostar_back"), "St-Pancras")
            + ". So the 121-min classes come from a dwell at the first call: counted from the "
            "listed arrival the ride is normal. The two links are in " + _usage(x, ids)
            + "; section 9 marks them (!), and the Paris-London lines there are wrong for "
            "the fastest rail. Open: take the arrival as the departure at a first call with "
            "a long dwell (or screen dwells over a few minutes), rebuild the network and "
            "rerun stage D.")


def rv_fast_links(x: dict, ctx: dict, out: dict) -> str:
    sus = ctx["suspect"]
    ids = [l for l in sus if l not in (5420, 5158)]
    p = x["params"]
    lyon = x["review"]["dwell"].get("lyon")
    lyon_txt = ""
    if lyon is not None and len(lyon):
        long_, floor = dwell_suspects(lyon)
        if len(long_):
            lyon_txt = (f" The Lyon class comes from {n_(long_['journeys'].sum())} journeys that "
                        f"dwell {int(long_['dwell'].min())} to {int(long_['dwell'].max())} min at "
                        f"Lyon Part-Dieu in {long_['feed_name'].nunique()} feeds ("
                        + ", ".join(sorted(long_["feed_name"].astype(str).unique()))
                        + f"): {int(long_['ride'].min())} to {int(long_['ride'].max())} min from "
                        f"the listed departure, {int(long_['ride_from_arr'].min())} to "
                        f"{int(long_['ride_from_arr'].max())} from the arrival, while journeys "
                        f"with a dwell under 15 min ride at least {floor:.0f} min "
                        "(data-upstream-hubs).")
    stale = ctx.get("suspect_stale", [])
    vmax = p.get("V_MAX_KMH") or {}
    return ("Other links faster than any real service. Checked now (ps-links): "
            + "; ".join(_link_text(x, l, speed=True) for l in ids) + "." + lyon_txt
            + " The others were not traced to a cause. All pass the straight-line speed "
            f"screens (V_MAX_KMH: rail {vmax.get('R')}, coach {vmax.get('B')} km/h), which do "
            "not depend on distance. With "
            "the two Eurostar links they are in " + _usage(x, list(sus))
            + "; section 9 marks them (!)."
            + (f" Entries no longer found in ps-links: {stale}." if stale else "")
            + " Open: a speed screen that depends on distance, the dwell rule of item "
            f"{item_no('eurostar')}, and a scan for such dwells.")


def rv_holes(x: dict, ctx: dict, out: dict) -> str:
    fuas, links, sparse, hubs = x["fuas"], x["links"], x["sparse"], x["hubs"]
    word = {"R": "rail", "B": "coach", "A": "air"}
    parts = []
    for a, b, modes in HOLE_PAIRS:
        o, d = _fua(a, fuas), _fua(b, fuas)
        txt = "; ".join(summary_text(direct_summary(links, sparse, hubs, o, d, m), word[m])
                        for m in modes)
        parts.append(f"{name_of(x, o)} -> {name_of(x, d)}: {txt}")
    o, d = _fua("Bilbao", fuas), _fua("Donostia", fuas)
    bd = direct_summary(links, sparse, hubs, o, d, "B")
    return ("Thin or missing direct services on well-known corridors, which the first "
            "version of section 9 showed without comment (direct links from the hubs of one "
            "FUA to those of the other; ps-links, ps-sparse): " + "; ".join(parts)
            + ". The plan's list of known "
            "holes (Verification 2) has only Amsterdam-Berlin, Madrid-Barcelona non-stop, "
            "Praha-Wien and Zürich-Milano; section 9 now flags both kinds. A thin class is "
            f"not always a hole: Bilbao -> Donostia has {bd['links']} coach links with "
            f"{bd['per_day']:g} departures a day in all, though its kept 1-a-day coach is "
            f"the fastest class (item {item_no('class_split')}). I compared nothing with "
            "public timetables (no network access). Open: an expected-service check for "
            "the well-known pairs, and more feeds (the review names Renfe and ALSA, the ÖBB "
            "through railjets, PKP international trains, KTEL, SAIS and Bulgarian coach "
            "operators).")


def _fua(name: str, fuas: pd.DataFrame) -> int:
    try:
        return V.resolve_fua(name, fuas)
    except ValueError:
        return V.resolve_fua(LOCAL_NAMES.get(name, name), fuas)


def rv_carriers(x: dict, ctx: dict, out: dict) -> str:
    c = x["review"]["carriers"]
    if c is None:
        return "Air carriers. data-upstream air-timetable could not be read; not checked now."
    hubs, links = x["hubs"], x["links"]
    parts = []
    for (s, t), rows in c["routes"].items():
        hs = hubs.loc[hubs["iata"].astype(str) == s, "hub"]
        ht = hubs.loc[hubs["iata"].astype(str) == t, "hub"]
        l = links[links["src"].isin(hs) & links["trg"].isin(ht) & (links["mode"] == "A")]
        f = f"{float(l['freq'].iloc[0]):g} a day (ps-links)" if len(l) else "no link"
        parts.append(f"{s} > {t} rows " + ", ".join(f"{k} {v:,}" for k, v in rows.items())
                     + f", {f}")
    none = [f"{CARRIERS_ABSENT[k]} ({k})" for k, v in c["absent"].items() if v == 0]
    some = [f"{CARRIERS_ABSENT[k]} ({k}) {v:,} rows" for k, v in c["absent"].items() if v]
    absent = ((", ".join(none[:-1]) + " and " + none[-1] if len(none) > 1 else none[0])
              + " have no row" if none else "") + ("; " + ", ".join(some) if some else "")
    return (f"Air carriers. The air data hold {len(c['codes'])} carrier codes (data-upstream "
            f"air-timetable: {', '.join(c['codes'])}). {absent}. Routes they fly (rows of "
            "air-timetable by carrier, and the frequency of the link): "
            + "; ".join(parts) + ". So flight frequencies, and "
            "with them the hidden wait, gtc and the gate, are low on routes these carriers "
            "fly. Open: state the carrier scope beside the comparison with the deck; flag "
            "such airport pairs or add the carriers.")


def rv_class_split(x: dict, ctx: dict, out: dict) -> str:
    alts, links, fuas = x["alts"], x["links"], x["fuas"]
    lk = links.set_index("link")
    dr = alts[alts["seq"].astype(str) == "R"]
    src = lk["src"].reindex(dr["l1"]).to_numpy()
    trg = lk["trg"].reindex(dr["l1"]).to_numpy()
    lf = lk["freq"].reindex(dr["l1"]).to_numpy(np.float64)
    pair_sum = links[links["mode"] == "R"].groupby(["src", "trg"])["freq"].sum()
    hs = pair_sum.reindex(list(zip(src, trg))).to_numpy(np.float64)
    share = lf / hs
    fast = (dr.assign(hs=hs).sort_values("T", kind="stable").groupby(["o", "d"]).head(1))
    o, d = _fua("Paris", fuas), _fua("Lyon", fuas)
    al = alts[(alts["o"] == o) & (alts["d"] == d) & (alts["seq"].astype(str) == "R")].sort_values("rank")
    ex = ""
    if len(al):
        r = al.iloc[0]
        l1 = int(r["l1"])
        same = links[(links["src"] == lk.at[l1, "src"]) & (links["trg"] == lk.at[l1, "trg"])
                     & (links["mode"] == "R")].sort_values("cls").head(3)
        f3 = float(same["freq"].sum())
        ex = (f" Example: Paris -> Lyon rank {int(r['rank'])} rides class "
              f"{int(lk.at[l1, 'cls'])} (link {l1}, {float(lk.at[l1, 'freq']):g} a day), while "
              f"classes {', '.join(str(int(c)) for c in same['cls'])} of "
              f"{x['names'][int(lk.at[l1, 'src'])]} > {x['names'][int(lk.at[l1, 'trg'])]} run "
              + " + ".join(f"{v:g}" for v in same["freq"]) + " a day; its freq_pooled is "
              f"{float(r['freq_pooled']):g}. Its hidden wait, {float(r['hwt']):.0f} min, would be "
              f"{min(600 / (2 * f3), 240):.0f} min at {f3:g} a day.")
    return ("A direct alternative carries the frequency of one riding-time class, not of "
            f"its hub pair. Of the {n_(len(dr))} kept direct rail alternatives, the median "
            f"share of the hub pair's departures (ps-links freq summed over its classes) is "
            f"{np.nanmedian(share):.2f}, and {pct((share < 0.5).sum(), len(share))} carry less "
            f"than half. The fastest direct rail of each pair runs a median "
            f"{fast['freq'].median():g} a day, its hub pair {np.nanmedian(fast['hs']):g}." + ex
            + " So freq/day and the hidden wait understate service on many corridors; section "
            "9 now shows pooled/day beside freq/day. Open (with Oded and Rajat): should the "
            "hidden wait of an alternative use the pooled departures of its members?")


def rv_near_ties(x: dict, ctx: dict, out: dict) -> str:
    p = x["params"]
    nt = near_ties(x["alts"], float(p["EPS_T_ABS"]))
    ex = ctx.get("near_tie_example", "")
    return ("Near-ties survive as separate alternatives. "
            f"{n_(nt['n'])} kept service alternatives lie within EPS_T_ABS = "
            f"{p['EPS_T_ABS']} min of a faster kept alternative of the same pair and mode "
            f"sequence, in {n_(nt['pairs'])} pairs; {n_(nt['same_ends'])} of them have the same "
            f"first and last hub as that one, and {n_(nt['within5'])} lie within 5 min "
            "(ps-alternatives). R1 opens a new leader more than eps_T after the last one, "
            "but representatives can lie closer than that, and R2 does not remove an "
            "alternative with as many transfers inside the band (rules.dominance needs "
            "T_p < T_q - eps_T)." + (f" Example: {ex}." if ex else "")
            + " R1 is meant to make such near-ties one alternative (plan, 'Choice "
            "alternative'). Open: merge representatives of one R1 key whose T lie within "
            "eps_T, or keep them and say so.")


def rv_distance(x: dict, ctx: dict, out: dict) -> str:
    dist = ctx["distance"]
    a, z = dist.iloc[0], dist.iloc[-1]
    ex = ctx.get("short_examples", "")
    return ("Set sizes fall as distance grows (section 6, table by distance band). "
            f"Under {DISTANCE_BANDS[1]} km: {n_(a['pairs'])} pairs, median "
            f"{a['median kept']:g} alternatives kept, the cap acts in {a['capped']} of them; "
            f"{a['2 x least T or more']} of their kept service alternatives take at least "
            f"twice the best service time and {a['with a flight']} fly. From "
            f"{DISTANCE_BANDS[-2]:,} km: median {z['median kept']:g}, capped {z['capped']}."
            + (f" {ex}" if ex else "")
            + " The cause is one alternative per mode class (R4) and R2 working only inside "
            "a class. Open (with Oded and Rajat): a detour or margin rule across classes, for "
            "example drop a class alternative whose T exceeds k times the pair's best T "
            "unless it is the only service class.")


def rv_car(x: dict, ctx: dict, out: dict) -> str:
    fi = ctx["car_fi"]
    alts = x["alts"]
    tmax = float(x["params"]["T_MAX_MIN"])
    ca = alts[alts["seq"].astype(str) == "C"]
    over = ca["T"] > tmax
    longest = ca.sort_values("T").iloc[-1]
    return (f"Car. (a) Finland: all {fi['n']} car rows from Finland to Sweden or Norway are "
            "marked fixed in ps-car, that is they cross the Gulf of Finland and go round the "
            f"Baltic Sea: {fi['min']:,.0f} to {fi['max']:,.0f} min; Oulu -> Stockholm "
            f"{fi['oulu_min']:,.0f} min over {fi['oulu_km']:,.0f} km, where the road through "
            "Tornio-Haparanda is about 1,000 km (the review's figure). The road network has "
            "no land border there; section 7 now says so. (b) The car has no 24-hour limit, "
            f"while every service path has T <= T_MAX_MIN = {tmax:g}: {n_(over.sum())} kept car "
            f"alternatives take longer ({n_((ca['T'] > 1.5 * tmax).sum())} more than "
            f"{1.5 * tmax / 60:g} h; the longest, {name_of(x, longest['o'])} -> "
            f"{name_of(x, longest['d'])}, {float(longest['T']) / 60:.1f} h); "
            f"{n_((over & (ca['rank'] == 1)).sum())} of them rank first and "
            f"{n_((over & (ca['rank'] <= 3)).sum())} third or better. Open: recompute the "
            "Finland rows on a network with the land border; apply T_MAX_MIN (or another "
            "cap) to the car, or flag such car alternatives.")


def rv_sea_access(x: dict, ctx: dict, out: dict) -> str:
    fuas, hubs, conn, alts = x["fuas"], x["hubs"], x["connectors"], x["alts"]
    h_id, t_id = _fua("Helsinki", fuas), _fua("Tallinn", fuas)
    hel = int(hubs.loc[hubs["iata"].astype(str) == "HEL", "hub"].iloc[0])
    tll = int(hubs.loc[hubs["iata"].astype(str) == "TLL", "hub"].iloc[0])
    c1 = conn[(conn["fua"] == h_id) & (conn["hub"] == tll)].iloc[0]
    c2 = conn[(conn["fua"] == t_id) & (conn["hub"] == hel)].iloc[0]
    u = (((alts["o"] == h_id) & (alts["h_first"] == tll)) | ((alts["o"] == t_id) & (alts["h_first"] == hel))
         | ((alts["d"] == h_id) & (alts["h_last"] == tll)) | ((alts["d"] == t_id) & (alts["h_last"] == hel)))
    car = x["car"]
    cross = f"{float(car.loc[(car['o'] == h_id) & (car['d'] == t_id), 'minutes'].iloc[0]):.1f}"
    return ("Fitted access over the sea. Access minutes fitted on straight-line km (plan, step "
            f"2) ignore water: Helsinki reaches Tallinn airport (TLL) in {float(c1['acc_min']):.1f} "
            f"min ({c1['source']}) and Tallinn reaches Helsinki-Vantaa in "
            f"{float(c2['acc_min']):.1f} min ({c2['source']}; ps-connectors), as if a road "
            f"crossed the Gulf of Finland; the crossing alone takes {cross} min (ps-car "
            "Helsinki -> Tallinn, item 2), before port drives, check-in and the wait for a "
            "sailing. "
            f"{n_(u.sum())} kept alternatives use these two connectors, "
            f"{n_((u & (alts['rank'] == 1)).sum())} at rank 1; "
            f"{n_(((alts['o'] == t_id) & (alts['h_first'] == hel)).sum())} are Tallinn "
            "alternatives that start at Helsinki-Vantaa. Open: no straight-line fit across "
            "the sea (route it through the crossing with boarding and sailings), and flag "
            "the two connectors.")


def rv_stations(x: dict, ctx: dict, out: dict) -> str:
    fuas, hubs, conn, links = x["fuas"], x["hubs"], x["connectors"], x["links"]
    word = {"R": "rail", "B": "coach"}
    st = []
    for n in STATION_FUAS:
        f = _fua(n, fuas)
        h = hubs[(hubs["fua"] == f) & (hubs["kind"] == "S")]
        parts = []
        for hub, nm in zip(h["hub"], h["name"].astype(str)):
            m = links.loc[(links["src"] == hub) | (links["trg"] == hub), "mode"].astype(str)
            vc = m.value_counts()
            parts.append(f"{nm} ("
                         + (", ".join(f"{word.get(k, k)} {int(v)}" for k, v in vc.items())
                            or "no link") + ")")
        st.append(f"{name_of(x, f)}: " + ", ".join(parts))
    cn = []
    for n, pat in CONNECTOR_CHECKS:
        f = _fua(n, fuas)
        c = conn[conn["fua"] == f].copy()
        c["name"] = [str(x["names"][int(h)]) for h in c["hub"]]
        c = c[c["name"].str.contains(pat, regex=True)].sort_values("egr_min")
        cn.append(f"{name_of(x, f)}: " + ", ".join(
            f"{r.name} {r.egr_min:.1f} ({r.source})" for r in c.itertuples()))
    return ("Main stations missing, and egress minutes that favour outer stations. Station "
            "hubs (ps-hubs, kind S) with the ps-links that start or end there: "
            + "; ".join(st) + ". So rail to Porto ends at Espinho (Campanhã is there only "
            "as the bus terminal), Warszawa's only rail hub is Zachodnia and Athina's is "
            "Acharnes (SKA). Egress minutes (ps-connectors, egr_min and source): "
            + "; ".join(cn) + ". With these, London -> Liverpool and Manchester -> Liverpool "
            "trains alight at Newton-le-Willows, and Kraków -> Katowice rail at Jaworzno "
            "Szczakowa (section 9 tables and ps-alternatives). Open: list the FUAs whose main "
            "station is missing from data-upstream-hubs, check the released minutes of "
            "London and Liverpool, and footnote the pairs affected.")


def rv_tz_coach(x: dict, ctx: dict, out: dict) -> str:
    links, hubs = x["links"], x["hubs"]
    tz = hubs.set_index("hub")["tz_std"]

    def pairs(mode):
        b = links[links["mode"] == mode].groupby(["src", "trg"])["ride"].min().rename("r").reset_index()
        q = b.merge(b.rename(columns={"src": "trg", "trg": "src", "r": "back"}), on=["src", "trg"])
        q["dz"] = tz.reindex(q["trg"]).to_numpy(np.float64) - tz.reindex(q["src"]).to_numpy(np.float64)
        q["diff"] = q["r"] - q["back"]
        return q
    b = pairs("B")
    east = b[np.isclose(b["dz"], 1)]
    same = b[np.isclose(b["dz"], 0)]
    r = pairs("R")
    r_east = r[np.isclose(r["dz"], 1)]
    k_e = int(east["diff"].between(-65, -55).sum())
    k_w = int(east["diff"].between(55, 65).sum())
    k_s = int(same["diff"].between(-65, -55).sum())
    return ("POSSIBLE: coach times across one time-zone step are off by an hour in some "
            "feeds. Coach hub pairs one zone apart with a link both ways (ps-links, fastest "
            f"ride each way; ps-hubs tz_std): {n_(len(east))}. Of these, {k_e} ride 55 to 65 "
            f"min faster eastbound than westbound ({pct(k_e, len(east))}), and {k_w} the "
            f"other way; among {n_(len(same))} same-zone pairs {k_s} "
            f"({pct(k_s, len(same), 2)}) show such a gap in one direction. Rail pairs one zone "
            f"apart: {int(r_east['diff'].between(-65, -55).sum())} of {len(r_east)}. Not "
            "proven: a route can differ by direction. An earlier audit of the time "
            "conversion (28 Sep) found a FlixBus feed an hour early in summer. Open: compare "
            "a few such lines with the raw feeds before relying on coach times across zones.")


def rv_rare(x: dict, ctx: dict, out: dict) -> str:
    alts, links = x["alts"], x["links"]
    lk = links.set_index("link")
    sv = alts[alts["seq"].astype(str) != "C"]
    ash = lk["active_share"].to_numpy(np.float64)
    ad = lk["active_dates"].to_numpy(np.float64)
    mn = np.full(len(sv), np.inf)
    md_ = np.full(len(sv), np.inf)
    for k in (1, 2, 3):
        l = sv[f"l{k}"].to_numpy(np.int64)
        ok = l >= 0
        s = np.where(ok, l, 0)
        mn = np.where(ok, np.minimum(mn, ash[s]), mn)
        md_ = np.where(ok, np.minimum(md_, ad[s]), md_)
    di = sv[sv["n"] == 0]
    dsh = ash[di["l1"].to_numpy(np.int64)]
    md_lim = int(x["params"]["MIN_COMMON_DATES"])
    return ("Rare and seasonal services read as daily. "
            f"{pct((sv['freq'] == 1).sum(), len(sv))} of the {n_(len(sv))} kept service "
            f"alternatives run exactly 1 a day (median {sv['freq'].median():g}, mean "
            f"{sv['freq'].mean():.2f}); {pct((mn < 0.3).sum(), len(sv))} ride a link active on "
            "fewer than 30% of the days of its own span (ps-links active_share), and "
            f"{pct((md_ < md_lim).sum(), len(sv))} a link active on fewer than {md_lim} dates. "
            f"Of the {n_(len(di))} direct alternatives, {n_((dsh < 0.3).sum())} ride a link "
            f"with active_share below 0.3, {n_(((dsh < 0.3) & (di['freq'].to_numpy() == 1)).sum())} "
            "of them at 1 a day. Under the median rule (decision 4) a weekly service and a "
            "daily one look the same in freq and in the hidden wait. Open: show the "
            "operating days (or freq_span) beside freq.")


def detours_text(x: dict, ctx: dict) -> str:
    """Kept service alternatives that are the only one of their class in the
    pair and either pass a hub that serves d before their last hub (an
    airport with egress to d: they overshoot and come back) or take twice the
    least T of the pair's kept service alternatives."""
    alts = x["alts"]
    svc = alts[alts["seq"].astype(str) != "C"].copy()
    n_cls = svc.groupby(["o", "d", "cls"])["rank"].transform("size")
    best_T = svc.groupby(["o", "d"])["T"].transform("min")
    single = n_cls == 1
    lk = x["links"].sort_values("link")
    h = V.link_hubs(svc[["o", "d", "seq", "rank", "l1", "l2", "l3", "h_last"]],
                    lk["src"].to_numpy(), lk["trg"].to_numpy())
    over = overshoots(h, x["connectors"])
    slow = (svc["T"] >= 2 * best_T).to_numpy()
    s = single.to_numpy()
    capped = (svc[["o", "d"]].merge(x["funnel"][["o", "d", "n_capped"]], on=["o", "d"],
                                    how="left")["n_capped"].to_numpy() > 0)
    txt = (f"of {n_(len(svc))} kept service alternatives, {n_(s.sum())} are the only one "
           "of their mode class in the pair: R2 works inside a class, so nothing removes "
           "them, and R4 shields them from the cap. "
           f"{n_((s & slow).sum())} of these take at least twice the least T of the pair's "
           f"kept service alternatives, {n_((s & slow & capped).sum())} of them in pairs "
           "where the cap acted. ")
    sa = x.get("sens_alts")
    if sa is not None:
        parts = []
        for o in sa["origin"].unique():
            a = sa[sa["origin"] == o]
            c = cap_detail(a[a["variant"] == "default"], a[a["variant"] == "no_cap"],
                           x["params"])
            parts.append(f"{name_of(x, o)} {n_(c['protected'])}")
        txt += ("Kept only because of R4 (beyond the cap in gtc order; no_cap variant of "
                "section 10): " + ", ".join(parts) + ". ")
    kinds = overshoot_kinds(h, x["connectors"], lk["mode"].astype(str).to_numpy())
    kv = kinds.value_counts()
    k_ex = {}
    for k in ("land_air", "board"):
        sel = h[(kinds == k).to_numpy()]
        if len(sel):
            r = with_routes(sel.sort_values(["rank", "o", "d"]).head(1), x).iloc[0]
            k_ex[k] = f"{name_of(x, r['o'])} -> {name_of(x, r['d'])} {r['seq']} '{r['route']}'"
    txt += (f"Next, {n_(over.sum())} kept service alternatives pass through an airport of "
            "d's catchment (finite egress to d in ps-connectors) before their last hub "
            f"({n_((s & over).sum())} of them alone in their class, {n_((s & slow & over).sum())} "
            "also among the slow ones above): "
            f"{n_(kv.get('land_ground', 0))} land there and ride on by rail or coach, for "
            "example into the city (Dublin > London Luton ~ Luton Airport Parkway > "
            "St-Pancras) or away and back (Amsterdam > Brussels (Zaventem) ~ Louvain > "
            "Bruxelles-Nord: Brussels airport's city is Leuven, the FUA with the shortest "
            f"access, so the transfer passes R0); {n_(kv.get('land_air', 0))} land there and "
            "fly on" + (f" ({k_ex['land_air']})" if "land_air" in k_ex else "")
            + f"; {n_(kv.get('board', 0))} reach it by ground and board a flight there"
            + (f" ({k_ex['board']})" if "board" in k_ex else "")
            + ". How many of the first kind go into the city and how many away and back was "
            "not counted.")
    return txt


def rail_replacement_text(x: dict, ctx: dict) -> str:
    m = x["members"].merge(x["links"][["link", "mode"]], on="link")
    rail_ag = set(m.loc[m["mode"] == "R", "agency"].astype(str))
    b = m[(m["mode"] == "B") & (m["route_type"] == 3) & m["agency"].astype(str).isin(rail_ag)]
    lks = set(b["link"].astype(int))
    alts = x["alts"]
    use = np.zeros(len(alts), dtype=bool)
    for k in (1, 2, 3):
        use |= alts[f"l{k}"].isin(lks).to_numpy()
    top = b.groupby(b["agency"].astype(str))["link"].nunique().sort_values(ascending=False).head(6)
    n_b = int((x["links"]["mode"] == "B").sum())
    return (f"{n_(len(lks))} of {n_(n_b)} coach links carry runs with route_type 3 of an "
            "agency that also runs rail links (ps-link-members), most by "
            + ", ".join(f"{a} ({v})" for a, v in top.items())
            + f"; {n_(use.sum())} kept alternatives use at least one of them. Not all "
            "are replacement buses (National Express and Vy run coaches as well).")


def oded_text(x: dict, ctx: dict) -> str:
    tr = x["trace"]
    r2 = tr[tr["rule"].astype(str) == "R2"]
    det = r2_details(r2["detail"])
    eps_abs = float(x["params"]["EPS_T_ABS"])
    od = oded_removals(det, eps_abs)
    one = od & (det["n_q"] - det["n_p"] == 1)
    later = (det["n_p"] < det["n_q"]) & (det["X_p"] > det["X_q"])
    txt = (f"of the {n_(len(det))} R2 removals in ps-trace (the ten named pairs, every R0 "
           f"candidate), {n_(later.sum())} are by a remover with fewer transfers that is "
           f"later; {n_(od.sum())} of these are later by more than EPS_T_ABS = {eps_abs:g} "
           f"min, so only the 5% term let them remove ({n_(one.sum())} with exactly one "
           f"transfer less, the shape counted in stage B). The largest band among the R2 "
           f"removals is {det['eps'].max():.1f} min; in principle it can reach 0.05 x "
           f"T_MAX_MIN = {0.05 * float(x['params']['T_MAX_MIN']):.0f} min.")
    runs = x["sens_runs"]
    if runs is not None:
        txt += (" On the two sensitivity origins (all destinations, default rules): "
                + oded_lesson(x, ctx))
    return txt


def earlier_text(x: dict, ctx: dict) -> str:
    moves, hubs, city = x["moves"], x["hubs"].set_index("hub"), x["city"]
    kind = hubs["kind"]
    a2s = moves[(kind.reindex(moves["hub_a"]).to_numpy() == "A")
                & (kind.reindex(moves["hub_b"]).to_numpy() == "S")]
    outside = int((city[a2s["hub_a"].to_numpy()] != city[a2s["hub_b"].to_numpy()]).sum())
    stn = hubs[(hubs["kind"] == "A") & (hubs["iata"].astype(str) == "STN")]
    stn_city = name_of(x, city[int(stn.index[0])]) if len(stn) else "n/a"
    tr = ctx.get("transfers", {})
    left = (f"ps-ftrans leaves out {n_(tr['left_out'])} of {n_(tr['rows'])} transfer rows "
            f"({pct(tr['left_out'], tr['rows'])}; manifest, transfers record)") if tr else ""
    return (f"Rechecked now: {outside} of {len(a2s)} airport-to-station moves end outside the "
            f"airport's city (ps-moves, cities as search.hub_cities); Stansted's city is "
            f"{stn_city}; {left}. Not rechecked tonight: season-scoped typical days; vehicle "
            "keys blocking about 0.3% of transfers; typical-day existence missing about 25% "
            "of date-exact chains.")


def new_findings(x: dict, ctx: dict) -> list:
    """Findings of this report that were not on the night's list."""
    alts, cp = x["alts"], ctx["common"]
    is_T = alts["basis"].astype(str) == "T"
    ground = alts["cls"].isin(["B", "R", "BR"])
    n_g = int((is_T & ground & cp["disjoint"]).sum())
    per = ctx["link_periods"]
    ac = ctx["air_coach"]
    st = seq_table(x["seq_counts"]).set_index("seq")
    rt = one_transfer_ratios(st).set_index("seq")["ratio"]
    out = [
        f"Typical days join ground links of different periods too: {n_(n_g)} kept "
        "ground-only alternatives (B, R, BR) on basis T have links with no common "
        "period (section 8). This comes from the typical-day fallback for links "
        f"sharing fewer than MIN_COMMON_DATES = {x['params'].get('MIN_COMMON_DATES')} dates, "
        "not from the air-ground rule.",
        f"All {n_(per['A']['n'])} air links end before 2026, while "
        f"{n_(per['B']['only_2026'])} of {n_(per['B']['n'])} coach links run only in 2026. "
        f"Of the {n_(ac['n'])} kept alternatives on typical days that use a flight and a "
        f"coach, {n_(ac['disjoint'])} ({pct(ac['disjoint'], ac['n'])}) have links that never "
        f"ran in a common period; {n_(ac['n'] - ac['disjoint'])} do (section 8).",
        f"The direct ground counts of the deck's Table 1 are close to our numbers of "
        f"ordered hub pairs (R: {n_(st.loc['R', 'n_r0'])} candidates on "
        f"{n_(ctx['hub_pairs']['R']['pairs'])} hub pairs, deck "
        f"{n_(DECK_TABLE1.set_index('seq').loc['R', 'paths'])}). POSSIBLE: the riding-time "
        "classes make our counts larger. For one transfer the gaps go both ways: the deck "
        "is larger for " + " and ".join(rt[rt <= 1].index) + ", ours for the other "
        f"{int((rt > 1).sum())} (AR {rt.get('AR', math.nan):.0f} times); no one cause fits "
        "both directions (section 5).",
    ]
    s, runs = x["sens"], x["sens_runs"]
    if s is None or runs is None:
        out.append("The sensitivity tables were not present when this report was written.")
        return out
    names = list(s["origin"].unique())

    def val(o, v, col):
        return s[(s["origin"] == o) & (s["variant"] == v)][col].iloc[0]

    out.append(
        "Under the plan's pooled gate the cap acts in every pair of both sensitivity "
        "origins: " + "; ".join(
            f"{name_of(x, o)} {n_(val(o, 'gate_pooled', 'pairs_capped'))} of "
            f"{n_(val(o, 'gate_pooled', 'pairs'))} pairs, {n_(val(o, 'gate_pooled', 'before_cap'))} "
            f"alternatives before the cap against {n_(val(o, 'default', 'before_cap'))} under "
            "top" for o in names) + " (section 10).")
    out.append(
        "The 5% term of eps_T decides about a tenth of the set: the flat band (eps_rel_0) "
        "keeps " + ", ".join(f"{name_of(x, o)} {pct(val(o, 'eps_rel_0', 'share_same_key'), 1, 1)}"
                             for o in names)
        + " of the default alternatives as they are (section 10). "
        + oded_lesson(x, ctx, short=True))
    sa = x.get("sens_alts")
    if sa is not None:
        mix = {o: no_r4_mix(sa, o) for o in names}
        out.append(
            "Comparing across classes without R4 leaves mainly the car and the direct "
            "flights: " + "; ".join(
                f"{name_of(x, o)} {m['n']:,} alternatives: {m['car']:,} car, {m['A']:,} direct "
                f"flights, {m['AA']:,} with two flights, {m['other']:,} others"
                for o, m in mix.items())
            + ". The car has no transfer and an infinite frequency, so it passes every gate.")
    return out


def build_report(x: dict, ctx: dict) -> str:
    """The markdown of the stage C/D report."""
    head = ["# Stages C and D of the path enumeration", "",
            f"Generated by pathset-stagecd.py on {dt.datetime.now():%Y-%m-%d %H:%M} from the "
            "stored tables of data-pathset (inputs read in "
            f"{x['load_s']:.0f} s). Plan, step 6: stage C = run time, memory, candidate "
            "counts by mode sequence against the deck's Table 1, cost of date-by-date "
            "chain counting, sensitivity to eps_T, mu and rho_F; stage D = the full path "
            "set. Numbers carry their source (table, column, file).", "",
            "T is clock minutes door to door, with fixed allowances: car access + TAU_DEP "
            "of the first mode + the riding minutes + the transfer minutes (a move between "
            "two hubs included) + TAU_ARR of the last mode + car egress (pathset/cost.py). "
            "TAU_DEP and TAU_ARR are "
            + ", ".join(f"{m} {x['params']['TAU_DEP'][m]} and {x['params']['TAU_ARR'][m]}"
                        for m in ("A", "R", "B"))
            + " min (A air, R rail, B coach), so a flight's T holds "
            f"{x['params']['TAU_DEP']['A'] + x['params']['TAU_ARR']['A']} min of airport time; "
            "the car has none. gtt is the deck's weighted minutes; gtc = gtt + hidden wait.",
            ""]
    sections = [section_run, section_stage_c, section_stage_d, section_funnel,
                section_sequences, section_sets, section_unserved, section_basis,
                section_pairs, section_sensitivity, section_open]
    body = [s(x, ctx) for s in sections]
    return "\n".join(head) + "\n" + "\n".join(body)


def prepare(x: dict) -> dict:
    """build_context plus the texts of the open items that need it."""
    ctx = build_context_base(x)
    ctx["detours"] = {"text": detours_text(x, ctx)}
    ctx["rail_replacement"] = rail_replacement_text(x, ctx)
    ctx["oded"] = oded_text(x, ctx)
    ctx["earlier"] = earlier_text(x, ctx)
    ctx["open"] = open_items(x, ctx)
    return ctx


def build_context_base(x: dict) -> dict:
    """``build_context`` without the open items (they need its results)."""
    ctx = {"timing": timing_table(x["attrs"]), "machine": machine_text()}
    p = x["params"]
    links = x["links"]
    run, log = x["run"], x["log"]
    ctx["wall"] = (log["seconds"] if log and log.get("seconds") is not None
                   else float(run["info"]["seconds"]))
    ctx["use"] = ctx["timing"]["total_s"].sum() / (int(run["info"]["workers"]) * ctx["wall"])
    rec = record_of(x["records"], "transfers")
    if rec:
        info = rec["info"]
        ctx["transfers"] = {"admissible": info["pairs"]["admissible"],
                            "rows": info["transfers"]["rows"],
                            "pairs_s": info["seconds"]["pairs_s"], "time": rec["time"],
                            "left_out": info["ftrans"]["transfers_left_out"]}
    seqs = DECK_TABLE1["seq"].tolist()
    best = x["best"][x["best"]["seq"].isin(seqs)]
    ctx["metrics_best"] = metrics_by_seq(path_metrics(best, links), seqs)
    kept = x["alts"][x["alts"]["seq"].isin(seqs)]
    ctx["metrics_kept"] = metrics_by_seq(path_metrics(kept, links), seqs)
    hp = {}
    for m in MODES:
        l = links[links["mode"] == m]
        g = l.groupby(["src", "trg"]).size()
        hp[m] = {"links": len(l), "pairs": len(g), "max": int(g.max())}
    ctx["hub_pairs"] = hp
    y26 = D.day_index(dt.date(2026, 1, 1))
    ctx["link_periods"] = {m: {"n": int((links["mode"] == m).sum()),
                               "only_2026": int(((links["mode"] == m)
                                                 & (links["first_day"] >= y26)).sum()),
                               "before_2026": int(((links["mode"] == m)
                                                   & (links["last_day"] < y26)).sum()),
                               "fixed_zero": int(((links["mode"] == m)
                                                  & (links["freq_fixed"] == 0)).sum())}
                           for m in MODES}
    hubs = x["hubs"]
    kind = hubs.set_index("hub")["kind"].astype(str)
    mv = x["moves"]
    ka = kind.reindex(mv["hub_a"]).to_numpy()
    kb = kind.reindex(mv["hub_b"]).to_numpy()
    a2s = mv[(ka == "A") & (kb == "S")]
    vc = a2s["kind"].astype(str).value_counts()
    ctx["air_station_moves"] = {"n": len(a2s), "road": int(vc.get("road", 0)),
                                "colocated": int(vc.get("colocated", 0))}
    ctx["git"] = git_info(paths=tuple(RUN_CODE))
    ctx["git_pathset"] = git_info(paths=("pathset",)).get("status")
    ctx["centres"] = fua_centres(x["fuas"])
    f0 = x["funnel"]
    ctx["distance"] = distance_table(f0, x["alts"], pair_km(f0["o"].to_numpy(),
                                                            f0["d"].to_numpy(), ctx["centres"]),
                                     DISTANCE_BANDS)
    ctx["suspect"], ctx["suspect_stale"] = suspect_links(links, x["names"])
    ctx["car_fi"] = car_finland(x)
    ctx["near_tie_example"] = near_tie_example(x)
    ctx["short_examples"] = short_examples(x)
    conn = x["connectors"].merge(hubs[["hub", "kind"]], on="hub")
    a = conn[(conn["kind"] == "A") & np.isfinite(conn["acc_min"].to_numpy(np.float64))]
    ctx["airport_access"] = {"n": len(a), "mean": float(a["acc_min"].mean())}
    s = hubs[hubs["kind"] == "S"]
    ctx["n_station_hubs"], ctx["n_station_fuas"] = len(s), int(s["fua"].nunique())
    f = x["funnel"]
    ctx["capped_not_at_cap"] = int(((f["n_capped"] > 0)
                                    & (f["n_kept"] != cap_of(f["n_classes"], p))).sum())
    t_ptr, t_out = load_transfer_csr(len(links), int(p["MIN_ACTIVE_DATES"]))
    lk = links.sort_values("link")
    ctx["reach"] = {(int(o), int(d)): reach_levels(int(o), int(d), x["connectors"], lk,
                                                   t_ptr, t_out)
                    for o, d in zip(x["unserved"]["o"], x["unserved"]["d"])}
    del t_ptr, t_out
    universe = search.fua_universe(x["connectors"], x["car"])
    ctx["car_cov"] = car_coverage(x["car"], universe)
    ctx["car_examples"] = car_missing_example(x)
    ctx["common"] = common_period(x["alts"], links)
    alts = x["alts"]
    is_T = (alts["basis"].astype(str) == "T").to_numpy()
    lm = links.sort_values("link")["mode"].astype(str).to_numpy()
    has = {m: np.zeros(len(alts), dtype=bool) for m in ("A", "B")}
    for k in (1, 2, 3):
        l = alts[f"l{k}"].to_numpy(np.int64)
        md_ = np.where(l >= 0, lm[np.where(l >= 0, l, 0)], "")
        for m in has:
            has[m] |= md_ == m
    ab = is_T & has["A"] & has["B"]
    ctx["air_coach"] = {"n": int(ab.sum()),
                        "disjoint": int((ab & ctx["common"]["disjoint"].to_numpy()).sum())}
    ctx["pairs"] = resolve_pairs(x["fuas"])
    ctx["basis_examples"] = basis_examples(x, ctx)
    return ctx


def car_finland(x: dict) -> dict:
    """Car rows from Finland to Sweden or Norway (ps-car, fuas.icc): rows,
    rows marked fixed, least, median and largest minutes, and Oulu ->
    Stockholm."""
    car, icc = x["car"], x["fuas"].set_index("id")["icc"].astype(str)
    oi = icc.reindex(car["o"]).to_numpy()
    di = icc.reindex(car["d"]).to_numpy()
    r = car[(oi == "FI") & np.isin(di, ["SE", "NO"])]
    out = {"n": len(r), "fixed": int(r["fixed"].astype(bool).sum()),
           "min": float(r["minutes"].min()), "median": float(r["minutes"].median()),
           "max": float(r["minutes"].max())}
    try:
        o, d = V.resolve_fua("Oulu", x["fuas"]), V.resolve_fua("Stockholm", x["fuas"])
        q = car[(car["o"] == o) & (car["d"] == d)].iloc[0]
        out.update(oulu_min=float(q["minutes"]), oulu_km=float(q["km"]))
    except (ValueError, IndexError):
        out.update(oulu_min=math.nan, oulu_km=math.nan)
    return out


def near_tie_example(x: dict) -> str:
    """The first well-known pair (in WELL_KNOWN_PAIRS order) with two kept
    alternatives of one mode sequence, the same first and last hub and T
    within EPS_T_ABS: names, ranks, links and T."""
    alts, eps = x["alts"], float(x["params"]["EPS_T_ABS"])
    for o, d, *_ in resolve_pairs(x["fuas"]):
        if o is None or d is None:
            continue
        a = alts[(alts["o"] == o) & (alts["d"] == d) & (alts["seq"].astype(str) != "C")]
        a = a.sort_values(["seq", "T"], kind="stable")
        for (s, hf, hl), g in a.groupby(["seq", "h_first", "h_last"], sort=False):
            if len(g) < 2:
                continue
            t = g["T"].to_numpy()
            i = np.flatnonzero(np.diff(t) <= eps)
            if len(i):
                p, q = g.iloc[i[0]], g.iloc[i[0] + 1]
                return (f"{name_of(x, o)} -> {name_of(x, d)} ranks {int(p['rank'])} and "
                        f"{int(q['rank'])}, both {s} {x['names'][int(hf)]} > "
                        f"{x['names'][int(hl)]} (links {int(p['l1'])} and {int(q['l1'])}, T "
                        f"{hm(p['T'])} and {hm(q['T'])})")
    return ""


SHORT_EXAMPLES = [("Rotterdam", "Gravenhage"), ("Manchester", "Liverpool"), ("Sofia", "Plovdiv")]


def short_examples(x: dict) -> str:
    """Kept alternatives of a few short pairs: how many, how many fly, the
    car and the fastest service time, the slowest flying one."""
    alts = x["alts"]
    out = []
    for a, b in SHORT_EXAMPLES:
        try:
            o, d = V.resolve_fua(a, x["fuas"]), V.resolve_fua(b, x["fuas"])
        except ValueError:
            continue
        al = alts[(alts["o"] == o) & (alts["d"] == d)]
        car = al.loc[al["seq"].astype(str) == "C", "T"]
        svc = al[al["seq"].astype(str) != "C"]
        fly = svc[svc["seq"].astype(str).str.contains("A")]
        if not len(svc):
            continue
        out.append(f"{name_of(x, o)} -> {name_of(x, d)} keeps {len(al)}"
                   + (f" (car {hm(car.iloc[0])}" if len(car) else " (no car")
                   + f", fastest service {hm(svc['T'].min())}), {len(fly)} of them fly"
                   + (f", the slowest of those in {hm(fly['T'].max())}" if len(fly) else ""))
    return ("Examples (ps-alternatives): " + "; ".join(out) + ".") if out else ""


def write_text(text: str, name: str) -> Path:
    """Write a text file inside PATHSET_DIR atomically (io.out_path guard)."""
    path = io.out_path(name)
    write_file(text, path)
    return path


def write_file(text: str, path: Path) -> Path:
    """Write a text file atomically (temporary name, then rename), UTF-8 with
    LF line ends. Used directly only for a report the caller places outside
    PATHSET_DIR (pathset-stagecd.py --out-dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)
    io.log(f"saved {path.name} in {path.parent}")
    return path
