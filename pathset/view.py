"""Viewer and stage B: the candidates of ONE FUA pair, their fate under the
rules and under variants of the rules, and the deck's top-5 selection.

Plan step 5 ("Viewer") and step 6, stage B: "you see the alternatives; side
by side: the deck's top 5 by weighted IVT per (hub OD pair, mode sequence) on
the same candidates, and the open choices" (plan, "Open with Oded and Rajat").

Everything here is a function of a candidate table (``pair_candidates``: the
PATH rows of every R0 candidate of the pair, search at MU = inf, with the
candidate id ``cand`` of ``search.decide`` and the board/alight hubs of each
link) and a parameter dict. Variants of the rules come from ``variants``;
``decide_variant`` applies one. The deck's components are those of
``pathset.cost`` (slide 229): t_a, t_dep, t_ivt, t_trans, t_tp, t_arr, t_e,
each a WEIGHTED term in minutes; gtt is their sum, T the clock time (the
same terms without weights), hwt the hidden wait and gtc = gtt + hwt.

The R0 clause PASSED_SHARE (search.passed_cities) is read when the network
is built; ``choose_passed_share`` picks it so that the viewer and stage B
enumerate the same R0 as the full run (finding of the 13 Nov review).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from . import cost as C
from . import io
from . import rules
from . import search

# --- Variants of the rules (stage B) -----------------------------------------
# Each variant: "order" of rules.apply_rules, parameter "params" replacing
# those of the run, and "freq": which frequency stands in for the path
# frequency (``with_frequency``). "deck" marks the deck's selection
# (``deck_top5``), which is not a rule variant.
RULE_VARIANTS = {
    "default": {},
    "october": {"order": "october"},
    "r2_weighted": {"params": {"R2_TIME": "weighted"}},
    "r1_class": {"params": {"R1_KEY": "class"}},
    "r1_pattern": {"params": {"R1_KEY": "pattern"}},
    "gate_path": {"params": {"R2_GATE": "path"}},
    "gate_pooled": {"params": {"R2_GATE": "pooled"}},
    "freq_fixed": {"freq": "fixed"},
    "freq_span": {"freq": "span"},
    "freq_first": {"freq": "first"},
    "deck_top5": {"deck": True},
}

# Sensitivity of stage B: parameter -> (key in pathset-params.yml, short
# name, values used when that key is absent). The fallback values are those
# of the plan ("Parameters": RHO_F 1 and 0.33 in sensitivity) and of the
# stage B task (EPS_T_ABS 10 and 20, MU 0.25 and 1.0), equal to the STAGE_B_*
# keys of pathset-params.yml.
SENSITIVITY = {
    "EPS_T_ABS": ("STAGE_B_EPS_T_ABS", "eps", (10, 20)),
    "MU": ("STAGE_B_MU", "mu", (0.25, 1.0)),
    "RHO_F": ("STAGE_B_RHO_F", "rho", (0.33, 1.0)),
}

# The deck's number of paths per (hub OD pair, mode sequence), slide 238;
# used when pathset-params.yml has no DECK_TOP_K
DECK_TOP_K = 5

COMPONENTS = list(C.COMPONENTS)
HUB_COLS = ["b1", "a1", "b2", "a2", "b3", "a3"]   # board / alight hub of link k
RIDE_COLS = ["ride1", "ride2", "ride3"]           # riding minutes of link k
FREQ_COLS = {"fixed": "f_fixed", "span": "f_span", "first": "f_first"}
RULE_ORDER = ["R5", "R2", "R2-cross", "R3"]       # order of the removed table


# --- FUA names ---------------------------------------------------------------
_FOLD = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l",
                       "đ": "d", "ı": "i", "þ": "th", "ð": "d"})


def normalise(text: str) -> str:
    """Case- and accent-free form of a name: lower case, accents dropped
    (NFKD, plus the letters NFKD keeps: ø, æ, ß, ł ...), every run of other
    characters one space."""
    s = unicodedata.normalize("NFKD", str(text).casefold().translate(_FOLD))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).translate(_FOLD)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s).split())


def load_fuas() -> pd.DataFrame:
    """FUA ids and names of data-upstream (``fuas``: id, name)."""
    return io.load("fuas", io.DATA, ["id", "name"])


def resolve_fua(text, fuas: pd.DataFrame | None = None) -> int:
    """FUA id from an id or a name, case and accent insensitive.

    Rule, in this order: digits are an id and must exist; a name equal to
    ``text`` after ``normalise``; a name with a part (split at '/') equal to
    it ("Brussel"); names containing it as whole words. One match is the
    answer; several raise ValueError listing them (an ambiguous name is
    refused), none raises ValueError. ``fuas``: id, name (default:
    ``load_fuas``).
    """
    fuas = load_fuas() if fuas is None else fuas
    ids = fuas["id"].to_numpy(np.int64)
    names = fuas["name"].astype(str).tolist()
    s = str(text).strip()
    if re.fullmatch(r"\d+", s):
        if int(s) not in set(ids.tolist()):
            raise ValueError(f"no FUA with id {s}")
        return int(s)
    q = normalise(s)
    if not q:
        raise ValueError(f"empty FUA name: {text!r}")
    norm = [normalise(n) for n in names]
    tests = (lambda n, raw: n == q,
             lambda n, raw: any(normalise(p) == q for p in raw.split("/")),
             lambda n, raw: f" {q} " in f" {n} ")
    for test in tests:
        hit = [i for i, (n, raw) in enumerate(zip(norm, names)) if test(n, raw)]
        if len(hit) == 1:
            return int(ids[hit[0]])
        if len(hit) > 1:
            listed = ", ".join(f"{names[i]} ({ids[i]})" for i in hit)
            raise ValueError(f"{text!r} is ambiguous: {listed}")
    raise ValueError(f"no FUA named {text!r}")


def fua_name(fuas: pd.DataFrame, fua: int) -> str:
    """Name of a FUA id (the id itself when unknown)."""
    hit = fuas.loc[fuas["id"].to_numpy(np.int64) == int(fua), "name"]
    return str(hit.iloc[0]) if len(hit) else str(fua)


def hub_names(hubs: pd.DataFrame) -> np.ndarray:
    """Name of every hub, indexed by hub id: the station hub's name, or the
    airport's name with its IATA code in brackets."""
    h = hubs.sort_values("hub", kind="stable")
    name = h["name"].astype(str).to_numpy(object)
    iata = h["iata"].fillna("").astype(str).to_numpy(object)
    air = (h["kind"].astype(str).to_numpy() == "A") & (iata != "")
    name[air] = [f"{n} ({c})" for n, c in zip(name[air], iata[air])]
    out = np.full(int(h["hub"].max()) + 1, "", dtype=object)
    out[h["hub"].to_numpy(np.int64)] = name
    return out


# --- R0 clause: PASSED_SHARE -------------------------------------------------
def manifest_records() -> list:
    """Records of PATHSET_DIR/ps-manifest.jsonl, oldest first ([] when the
    file is missing)."""
    path = io.out_path("ps-manifest.jsonl")
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_record(records: list, step: str, key: str | None = None) -> dict | None:
    """The latest manifest record of ``step`` (with ``key`` in its info when
    given), or None."""
    for rec in reversed(records):
        if rec.get("step") == step and (key is None or key in rec.get("info", {})):
            return rec
    return None


def choose_passed_share(cli, params: dict, records: list) -> tuple:
    """(PASSED_SHARE, source) for the viewer and stage B.

    Rule: the command-line value; else PASSED_SHARE of the parameters
    (pathset-params.yml); else the passed_share of the latest
    ``enumerate`` record of the manifest, so that the R0 of the viewer equals
    the R0 of ps-funnel and ps-trace; else search.passed_share's default (1).
    pathset-enumerate.py records the value it ran with (info passed_share).
    """
    if cli is not None:
        return search.passed_share({"PASSED_SHARE": float(cli)}), "command line"
    if params.get("PASSED_SHARE") is not None:
        return search.passed_share(params), "pathset-params.yml"
    rec = last_record(records, "enumerate", "passed_share")
    if rec is not None:
        share = search.passed_share({"PASSED_SHARE": rec["info"]["passed_share"]})
        return share, f"full run of {rec['time']} (ps-manifest)"
    return search.passed_share({}), "default of search.passed_share"


# --- Candidates of one pair --------------------------------------------------
def link_hubs(c: pd.DataFrame, src: np.ndarray, trg: np.ndarray) -> pd.DataFrame:
    """Board and alight hub of every link of the paths (``HUB_COLS``; -1 for
    an unused link and for the car)."""
    out = c.copy()
    for k in (1, 2, 3):
        link = out[f"l{k}"].to_numpy(np.int64)
        used = link >= 0
        safe = np.where(used, link, 0)
        out[f"b{k}"] = np.where(used, np.asarray(src)[safe], -1).astype(np.int64)
        out[f"a{k}"] = np.where(used, np.asarray(trg)[safe], -1).astype(np.int64)
    return out


def _link_matrix(c: pd.DataFrame, links: pd.DataFrame):
    """(ps-links sorted by link, link ids (N, 3), used mask, safe ids)."""
    lk = links.sort_values("link", kind="stable")
    if not (lk["link"].to_numpy() == np.arange(len(lk))).all():
        raise ValueError("ps-links.link must be 0 .. L-1")
    ls = np.column_stack([c[f"l{k}"].to_numpy(np.int64) for k in (1, 2, 3)])
    used = ls >= 0
    return lk, ls, used, np.where(used, ls, 0)


def link_freqs(c: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Frequencies of the stage B variants (6): f_fixed and f_span, the least
    freq_fixed and freq_span of the path's links (ps-links; deck slide 237
    for the fixed denominator), and f_first, the first link's freq (the
    deck's hidden-wait form, slide 228). Unused links do not count; the car
    keeps inf."""
    lk, _, used, safe = _link_matrix(c, links)
    out = c.copy()
    for col, src in (("f_fixed", "freq_fixed"), ("f_span", "freq_span")):
        v = lk[src].to_numpy(np.float64)[safe]
        out[col] = np.where(used, v, np.inf).min(axis=1)
    out["f_first"] = np.where(used[:, 0], lk["freq"].to_numpy(np.float64)[safe[:, 0]],
                              np.inf)
    return out


def link_rides(c: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Riding minutes of every link of the paths (``RIDE_COLS``: ps-links
    ride, the mean of the riding-time class; NaN for an unused link and the
    car). Two paths with the same hubs differ in these when they use
    different riding-time classes (plan, "Service link")."""
    lk, _, used, safe = _link_matrix(c, links)
    v = np.where(used, lk["ride"].to_numpy(np.float64)[safe], np.nan)
    out = c.copy()
    for k, col in enumerate(RIDE_COLS):
        out[col] = v[:, k]
    return out


def number(cands: pd.DataFrame) -> pd.DataFrame:
    """Candidates in the order of ``search.decide`` (increasing l1, l2, l3,
    the car first) with ``cand`` = position, so that every variant numbers
    them alike."""
    c = cands.sort_values(["l1", "l2", "l3"], kind="stable").reset_index(drop=True)
    c["cand"] = np.arange(len(c), dtype=np.int64)
    return c


def pair_candidates(net, o: int, d: int, params: dict | None = None,
                    links: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every R0 candidate of the pair (o, d) with the PATH columns
    (``search.candidates`` at MU = inf, the car included), numbered as
    ``search.decide`` numbers them (``cand``), with the board/alight hubs of
    its links (``HUB_COLS``) and, when ``links`` (ps-links with link, freq,
    freq_fixed, freq_span[, ride]) is given, the frequencies of
    ``link_freqs`` and, with ride, the riding minutes of ``link_rides``.
    The R0 clause PASSED_SHARE is that of the network ``net``."""
    params = io.P if params is None else params
    c = number(search.candidates(net, int(o), [int(d)], params, math.inf))
    c = link_hubs(c, net.arrays.src, net.arrays.trg)
    if links is None:
        return c
    c = link_freqs(c, links)
    return link_rides(c, links) if "ride" in links.columns else c


# --- Variants ----------------------------------------------------------------
def sensitivity_values(params: dict) -> dict:
    """Parameter -> values of the stage B sensitivity: STAGE_B_<name> of the
    parameters, else the fallback of ``SENSITIVITY``."""
    out = {}
    for name, (key, _, fallback) in SENSITIVITY.items():
        vals = params.get(key)
        out[name] = tuple(fallback if vals is None else vals)
    return out


def missing_parameters(params: dict) -> list:
    """Keys this module reads with a fallback because pathset-params.yml
    lacks them."""
    keys = [k for k, _, _ in SENSITIVITY.values()] + ["DECK_TOP_K", "PASSED_SHARE"]
    return [k for k in keys if params.get(k) is None]


def variants(params: dict | None = None) -> dict:
    """Name -> spec of every stage B variant: ``RULE_VARIANTS`` and one
    variant per sensitivity value ("eps_10", "mu_0.25", ...)."""
    params = io.P if params is None else params
    out = dict(RULE_VARIANTS)
    for name, vals in sensitivity_values(params).items():
        short = SENSITIVITY[name][1]
        for v in vals:
            out[f"{short}_{v}"] = {"params": {name: v}}
    return out


def variant_notes(params: dict | None = None) -> dict:
    """Name -> what the variant changes, in plain words, with the values of
    ``params`` (report of stage B)."""
    p = io.P if params is None else params
    notes = {
        "default": (
            f"R3 first: a candidate is cut only if its gtt and its gtc both "
            f"exceed (1 + MU) x the least of its mode class (MU {p['MU']}). Then R1 "
            f"by {p['R1_KEY']} (R1_KEY): in increasing T a candidate joins the "
            f"first leader whose T is within eps_T = max({p['EPS_T_ABS']} min, "
            f"{p['EPS_T_REL']} x least T of the class); the representative is the "
            f"fastest member with freq >= RHO_F ({p['RHO_F']}) x the group's largest. "
            f"R2 on (transfers, {p['R2_TIME']} time) inside a mode class: p removes q "
            f"only if freq(p) >= RHO_F x "
            + {"top": "the largest path frequency among the members of q's group ",
               "path": "freq(q), the path frequency of q's representative ",
               "pooled": "the pooled frequency of q's group (distinct first links) "
               }[p["R2_GATE"]]
            + f"(R2_GATE {p['R2_GATE']}). R4 keeps the last alternative of a class; "
            f"R5 caps at "
            f"max(K_CAP {p['K_CAP']}, classes present + K_EXTRA {p['K_EXTRA']}) by "
            f"gtc (plan step 4)."),
        "october": ("Order of the 16 Oct note: R1 and R2 on all candidates, then the "
                    "margin on gtc alone among the kept alternatives of each class, "
                    "then the cap."),
        "r2_weighted": ("R2 compares the weighted time gtt instead of the clock time "
                        f"T (29 Oct report). A transfer costs TAU_TP ({p['TAU_TP']} min) "
                        f"plus BETA_TRANS ({p['BETA_TRANS']}) times its minutes, so a "
                        "path with one transfer more must be clearly faster on the "
                        "clock to stay."),
        "r1_class": ("R1 keyed on the mode class and the transfer count instead of "
                     "the mode sequence: rail-coach and coach-rail near-ties merge."),
        "r1_pattern": ("R1 keyed on the mode sequence and the route pattern (city "
                       "sequence): near-ties through different transfer cities stay "
                       "apart, as in the thesis example (corridor merge off)."),
        "gate_path": ("R2_GATE = path: p removes q only if freq(p) >= RHO_F x the "
                      "path frequency of q's representative. The representative "
                      "may itself run only RHO_F times as often as the most "
                      "frequent member of its group, so a remover can run 1/RHO_F^2 "
                      "times less often than a member it removes."),
        "gate_pooled": ("R2_GATE = pooled, the plan's wording: p removes q only if "
                        "freq(p) >= RHO_F x the pooled frequency of q's R1 group "
                        "(sum over its distinct first links of the largest member "
                        "frequency). Groups pool feeders from many stations that "
                        "converge on a few trunk trains, so R2 removes about half "
                        "the alternatives and the cap R5 chooses most of the set."),
        "freq_fixed": ("Path frequency replaced by the least freq_fixed of its links "
                       f"(departures from {p['FIXED_FREQ_START']} to "
                       f"{p['FIXED_FREQ_END']} / days of that period; the deck's fixed "
                       "denominator, slide 237); hwt and gtc recomputed. A link "
                       "without departures in that period gets 0 (most coach links "
                       "and many rail links: their feeds start in 2026), so its paths "
                       f"get the largest hidden wait (HWT_MAX {p['HWT_MAX']} min), "
                       "cannot remove a path with departures through the gate, and "
                       "are removed by any noticeably better path of the class, "
                       "since 0 >= RHO_F x 0."),
        "freq_span": ("Path frequency replaced by the least freq_span of its links "
                      "(departures / days from the link's first to its last date); "
                      "hwt and gtc recomputed."),
        "freq_first": ("Path frequency replaced by the frequency of the first link "
                       "(the deck's hidden-wait form, hwt_first, slide 228); hwt and "
                       "gtc recomputed."),
        "deck_top5": (f"No rules: per (first hub, last hub, mode sequence) the "
                      f"{int(p.get('DECK_TOP_K') or DECK_TOP_K)} candidates with the "
                      "least t_ivt (weighted riding time, no transfer time; deck "
                      "slide 238). The car is its own group."),
    }
    labels = {"EPS_T_ABS": "Time band EPS_T_ABS = {v} min instead of {b}.",
              "MU": "R3 margin MU = {v} instead of {b}.",
              "RHO_F": "Frequency gate RHO_F = {v} instead of {b}."}
    for name, vals in sensitivity_values(p).items():
        for v in vals:
            notes[f"{SENSITIVITY[name][1]}_{v}"] = labels[name].format(v=v, b=p[name])
    return notes


def variant_spec(variant, params: dict | None = None) -> dict:
    """The spec of a variant: a name of ``variants(params)`` or a dict of the
    same form (order, params, freq, deck)."""
    if isinstance(variant, dict):
        return variant
    known = variants(params)
    if variant not in known:
        raise ValueError(f"unknown variant {variant!r}; one of {list(known)}")
    return known[variant]


def with_frequency(cands: pd.DataFrame, params: dict, how: str = "path") -> pd.DataFrame:
    """The candidates with ``freq``, ``hwt`` and ``gtc`` recomputed at another
    frequency: "path" leaves them as they are; "fixed", "span", "first" put
    f_fixed, f_span or f_first (``link_freqs``) in place of the path
    frequency, hwt = cost.hidden_wait at it (min(BETA_HWT / (2 f), HWT_MAX),
    deck slide 228) and gtc = gtt + hwt. The car keeps hwt 0 and freq inf."""
    if how == "path":
        return cands
    if how not in FREQ_COLS:
        raise ValueError(f"frequency must be path or one of {list(FREQ_COLS)}")
    col = FREQ_COLS[how]
    if col not in cands.columns:
        raise ValueError(f"column {col} missing: pass ps-links to pair_candidates")
    out = cands.copy()
    f = out[col].to_numpy(np.float64)
    car = (out["seq"].astype(str) == "C").to_numpy()
    hwt = np.where(car, 0.0, C.hidden_wait(f, C.param_array(params)))
    out["freq"] = np.where(car, np.inf, f).astype(np.float32)
    out["hwt"] = hwt.astype(np.float32)
    out["gtc"] = (out["gtt"].to_numpy(np.float64) + hwt).astype(np.float32)
    return out


def deck_top5(cands: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """The deck's selection on the same candidates (slide 238): per
    (h_first, h_last, seq) the DECK_TOP_K (default 5, the deck's number) with
    the least t_ivt (weighted riding time; no transfer time). Ties: lower
    gtt, then lower cand. The car (h_first = h_last = -1, seq 'C') is a group
    of its own. Returns the selected rows in increasing (h_first, h_last,
    seq, deck_rank) with the column deck_rank (1..)."""
    params = io.P if params is None else params
    k = int(params.get("DECK_TOP_K") or DECK_TOP_K)
    c = cands.sort_values(["h_first", "h_last", "seq", "t_ivt", "gtt", "cand"],
                          kind="stable")
    rank = c.groupby(["h_first", "h_last", "seq"], sort=False).cumcount() + 1
    out = c[rank.to_numpy() <= k].copy()
    out["deck_rank"] = rank[rank.to_numpy() <= k].to_numpy().astype(np.int16)
    return out.reset_index(drop=True)


def selection_alternatives(sel: pd.DataFrame) -> pd.DataFrame:
    """A selection without rules (``deck_top5``) in the form of
    rules.ALTERNATIVES: every row its own alternative, in increasing (gtc,
    cand)."""
    s = sel.sort_values(["gtc", "cand"], kind="stable")
    cls = [rules.mode_class(x) for x in s["seq"].astype(str)]
    return pd.DataFrame({
        "alt": s["cand"].to_numpy(np.int64), "seq": s["seq"].to_numpy(), "cls": cls,
        "n": s["n"].to_numpy(), "T": s["T"].to_numpy(), "gtt": s["gtt"].to_numpy(),
        "gtc": s["gtc"].to_numpy(), "freq": s["freq"].to_numpy(),
        "freq_pooled": s["freq"].to_numpy(), "M": np.nan, "n_variants": 0,
        "members": [[int(x)] for x in s["cand"]], "first_link": s["l1"].to_numpy(),
        "key": s["key"].to_numpy(), "cities": s["cities"].to_numpy(),
        "eps_T": np.nan, "status": "selected"})


def decide_variant(cands: pd.DataFrame, params: dict | None = None,
                   variant="default"):
    """(alternatives, trace) of the pair under a variant of the rules.

    The variant's params replace those of ``params``; its frequency is put in
    place of the path frequency (``with_frequency``); then
    rules.apply_rules with its order and R4_GUARANTEE, on the candidates
    numbered by ``cand``. For "deck_top5" the alternatives are the deck's
    selection (``selection_alternatives``) and the trace marks every
    candidate selected or not. Columns as rules.ALTERNATIVES and rules.TRACE.
    """
    params = io.P if params is None else params
    spec = variant_spec(variant, params)
    p = {**params, **spec.get("params", {})}
    if spec.get("deck"):
        sel = deck_top5(cands, p)
        alts = selection_alternatives(sel)
        chosen = np.isin(cands["cand"].to_numpy(), alts["alt"].to_numpy())
        trace = pd.DataFrame({
            "cand": cands["cand"].to_numpy(np.int64), "seq": cands["seq"].to_numpy(),
            "cls": [rules.mode_class(x) for x in cands["seq"].astype(str)],
            "alt": np.where(chosen, cands["cand"].to_numpy(np.int64), -1),
            "status": np.where(chosen, "selected", "not_selected"),
            "rule": "deck_top5", "by": -1, "detail": ""})
        return alts, trace.sort_values("cand", kind="stable").reset_index(drop=True)
    c = with_frequency(cands, p, spec.get("freq", "path"))
    inp = pd.DataFrame({"cand": c["cand"], "seq": c["seq"], "n": c["n"],
                        "T": c["T"], "gtt": c["gtt"], "gtc": c["gtc"],
                        "freq": c["freq"], "first_link": c["l1"], "key": c["key"],
                        "cities": c["cities"]})
    return rules.apply_rules(inp, p, order=spec.get("order", "r3_first"),
                             guarantee=bool(p["R4_GUARANTEE"]))


def trace_rows(cands: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """ps-trace rows of the pair (contract TRACE) from the candidates and the
    trace of the default rules."""
    c = cands.sort_values("cand", kind="stable").reset_index(drop=True)
    return search.as_table(search.trace_table(c, trace), "trace")


# --- Reading the outcome -----------------------------------------------------
def route(row, names) -> str:
    """Hub names of a path joined by ' > ' for a ride and ' ~ ' for a move
    between two hubs of a transfer; 'car' for the car. ``row`` has the
    columns HUB_COLS (``link_hubs``), ``names`` maps hub id to name."""
    if str(row["seq"]) == "C":
        return "car"
    parts, last = [], -1
    for k in (1, 2, 3):
        b, a = int(row[f"b{k}"]), int(row[f"a{k}"])
        if b < 0:
            break
        if last < 0:
            parts.append(str(names[b]))
        elif b != last:
            parts.append(" ~ " + str(names[b]))
        parts.append(" > " + str(names[a]))
        last = a
    return "".join(parts)


def rides(row) -> str:
    """Riding minutes of the path's links joined by '+' ('19+159'), from
    ``RIDE_COLS``; '' for the car or without those columns."""
    vals = [row[k] for k in RIDE_COLS if k in row]
    return "+".join(f"{float(v):.0f}" for v in vals if np.isfinite(float(v)))


def funnel(trace: pd.DataFrame, alts: pd.DataFrame, order: str = "r3_first") -> list:
    """Counts after each step, in the order the steps ran: [(step, n)].

    R0: candidates; R3: not cut by the margin (r3_first: candidates; october:
    alternatives); R1: alternatives (groups); R2: alternatives not removed by
    R2 or by the comparison across classes; cap: the kept set.
    """
    status = trace["status"].astype(str)
    reps = trace[trace["cand"].to_numpy() == trace["alt"].to_numpy()]
    n_alt = len(reps)
    n_removed = int(reps["rule"].astype(str).isin(["R2", "R2-cross"]).sum())
    n_cut_alt = int((reps["status"].astype(str) == "cut_r3").sum())
    if order == "october":
        return [("R0", len(trace)), ("R1", n_alt), ("R2", n_alt - n_removed),
                ("R3", n_alt - n_removed - n_cut_alt), ("cap", len(alts))]
    return [("R0", len(trace)), ("R3", int((status != "cut_r3").sum())),
            ("R1", n_alt), ("R2", n_alt - n_removed), ("cap", len(alts))]


def alternatives_table(cands: pd.DataFrame, alts: pd.DataFrame, names) -> pd.DataFrame:
    """The kept alternatives for printing: rank, seq, route, rides (riding
    minutes per link, when ``cands`` has RIDE_COLS), the deck's components
    (weighted minutes, summing to gtt), T (clock minutes), gtt, hwt, gtc,
    freq (of the alternative as the rules saw it), freq_pooled, n_variants,
    status, key (link ids), cities, cand."""
    c = cands.set_index("cand", drop=False)
    rows = c.loc[alts["alt"].to_numpy(np.int64)].reset_index(drop=True)
    out = pd.DataFrame({"rank": np.arange(1, len(alts) + 1), "seq": rows["seq"].to_numpy(),
                        "route": [route(r, names) for _, r in rows.iterrows()],
                        "rides": [rides(r) for _, r in rows.iterrows()]})
    for k in COMPONENTS + ["T", "gtt"]:
        out[k] = rows[k].to_numpy(np.float64)
    out["gtc"] = alts["gtc"].to_numpy(np.float64)
    out["hwt"] = out["gtc"] - out["gtt"]
    out["freq"] = alts["freq"].to_numpy(np.float64)
    out["freq_pooled"] = alts["freq_pooled"].to_numpy(np.float64)
    out["n_variants"] = alts["n_variants"].to_numpy()
    out["status"] = alts["status"].to_numpy()
    out["key"] = rows["key"].to_numpy()
    out["cities"] = rows["cities"].to_numpy()
    out["cand"] = rows["cand"].to_numpy()
    return out[["rank", "seq", "route", "rides"] + COMPONENTS
               + ["T", "gtt", "hwt", "gtc", "freq", "freq_pooled", "n_variants",
                  "status", "key", "cities", "cand"]]


def removed_table(cands: pd.DataFrame, trace: pd.DataFrame, names) -> pd.DataFrame:
    """Alternatives removed after R1 (the cap R5, R2, the comparison across
    classes, the margin of the October order): their route, rule, the
    remover's route and the numbers of the test. Ordered by rule
    (``RULE_ORDER``), then in increasing gtc: for the cap this is the cap
    rank, so the first rows are the alternatives the cap dropped first
    after the kept set; for R2 the removed alternatives with the least gtc.
    ``cands`` should carry the frequency the rules saw (``with_frequency``)."""
    reps = trace[(trace["cand"].to_numpy() == trace["alt"].to_numpy())
                 & ~trace["status"].astype(str).isin(rules.KEPT)]
    cols = ["seq", "route", "T", "gtt", "gtc", "freq", "rule", "by_route", "detail",
            "cand"]
    if not len(reps):
        return pd.DataFrame(columns=cols)
    c = cands.set_index("cand", drop=False)
    rows = c.loc[reps["cand"].to_numpy(np.int64)].reset_index(drop=True)
    by = reps["by"].to_numpy(np.int64)
    out = pd.DataFrame({
        "seq": rows["seq"].to_numpy(),
        "route": [route(r, names) for _, r in rows.iterrows()],
        "T": rows["T"].to_numpy(np.float64), "gtt": rows["gtt"].to_numpy(np.float64),
        "gtc": rows["gtc"].to_numpy(np.float64), "freq": rows["freq"].to_numpy(np.float64),
        "rule": reps["rule"].astype(str).to_numpy(),
        "by_route": [route(c.loc[b], names) if b >= 0 else "" for b in by],
        "detail": reps["detail"].astype(str).to_numpy(),
        "cand": rows["cand"].to_numpy()})
    out["_o"] = [RULE_ORDER.index(r) if r in RULE_ORDER else len(RULE_ORDER)
                 for r in out["rule"]]
    return (out.sort_values(["_o", "gtc", "cand"], kind="stable").drop(columns="_o")
            .reset_index(drop=True))[cols]


def removed_head(rem: pd.DataFrame, per_rule: int) -> pd.DataFrame:
    """The first ``per_rule`` rows of every rule of ``removed_table``, in its
    order, so that a long cap list does not hide the R2 removals."""
    if not len(rem):
        return rem
    return rem.groupby("rule", sort=False).head(int(per_rule)).reset_index(drop=True)


def rule_counts(rem: pd.DataFrame) -> str:
    """Removed alternatives per rule, e.g. 'R5 61, R2 137'."""
    if not len(rem):
        return ""
    n = rem["rule"].value_counts(sort=False)
    return ", ".join(f"{r} {n[r]}" for r in rem["rule"].unique())


def r3_summary(cands: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """Candidates cut by R3 per mode sequence: how many, and the least T, gtt
    and gtc among them."""
    cut = trace["status"].astype(str).to_numpy() == "cut_r3"
    c = cands.set_index("cand").loc[trace["cand"].to_numpy(np.int64)[cut]]
    if not len(c):
        return pd.DataFrame(columns=["seq", "n_cut", "min_T", "min_gtt", "min_gtc"])
    g = c.groupby("seq", sort=True)
    return pd.DataFrame({"n_cut": g.size(), "min_T": g["T"].min(),
                         "min_gtt": g["gtt"].min(),
                         "min_gtc": g["gtc"].min()}).reset_index()


def members(alts: pd.DataFrame) -> set:
    """Candidate ids of all members of the alternatives."""
    return {int(m) for ms in alts["members"] for m in ms}


def compare(base: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Overlap of a variant's alternatives (``other``) with the default ones
    (``base``), both as rules.ALTERNATIVES.

    n_same: alternatives with the same representative; n_base_found: default
    alternatives whose representative is a member of some alternative of the
    variant; n_other_found: alternatives of the variant whose representative
    is a member of some default alternative.
    """
    b_alt = set(base["alt"].astype(int))
    o_alt = set(other["alt"].astype(int))
    return {"n_same": len(b_alt & o_alt),
            "n_base_found": len(b_alt & members(other)),
            "n_other_found": len(o_alt & members(base))}


def seq_summary(alts: pd.DataFrame) -> str:
    """Mode sequences of the alternatives with their counts, e.g. 'R:2 RR:1'."""
    if not len(alts):
        return ""
    n = alts["seq"].astype(str).value_counts().sort_index()
    return " ".join(f"{s}:{k}" for s, k in n.items())


def fmt_table(df: pd.DataFrame, digits: int = 1) -> str:
    """Plain text table, floats rounded."""
    out = df.copy()
    for k in out.columns:
        if pd.api.types.is_float_dtype(out[k]):
            out[k] = out[k].astype(np.float64).round(digits)
    return out.to_string(index=False)


def md_table(df: pd.DataFrame, digits: int = 1) -> str:
    """Markdown table, floats rounded; '|' in cells escaped."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else ("inf" if np.isinf(v) else f"{v:.{digits}f}")
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)
