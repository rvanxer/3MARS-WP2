"""Transfer table of every admissible pair of service links (plan step 2,
"Transfer table").

Reads ps-hubs, ps-links, ps-events, ps-event-masks.npy, ps-link-members,
ps-moves, ps-connectors. Writes ps-transfers and ps-ftrans (contract:
pathset/schemas.py) and one record in ps-manifest.jsonl. The pair rules are
in pathset/pairs.py, the counting in pathset/transfer.py (its "Consumers"
section gives the call sequence used here).

Run:  python -X utf8 pathset-transfers.py
"""

#%% Imports
import time

import numpy as np

from pathset import connect as C
from pathset import io
from pathset import pairs as PR
from pathset import transfer as T

P = io.P
INPUTS = ["ps-hubs.parquet", "ps-links.parquet", "ps-events.parquet",
          "ps-event-masks.npy", "ps-link-members.parquet", "ps-moves.parquet",
          "ps-connectors.parquet"]


#%% Inputs
def read_inputs() -> dict:
    """The network tables; the masks without a memory map (a map would block
    a later atomic rewrite of the file on Windows)."""
    return {
        "hubs": io.load_out("ps-hubs"),
        "links": io.load_out("ps-links"),
        "events": io.load_out("ps-events", ["dep", "ride", "trip"]),
        "masks": io.load_npy("ps-event-masks"),
        "members": io.load_out("ps-link-members", ["link", "agency"]),
        "moves": io.load_out("ps-moves"),
        "connectors": io.load_out("ps-connectors"),
    }


#%% a. Admissible pairs
def step_pairs(x: dict):
    city = C.hub_city(x["hubs"], x["connectors"])
    pairs, dropped = PR.admissible_pairs(x["links"], x["moves"], city,
                                         P["WINDOWS"])
    io.log("a. pairs: " + "; ".join(f"{k} {v:,}" for k, v in dropped.items())
           + f"; admissible {len(pairs):,} ({PR.memory_mb(pairs):,.0f} MB)")
    table = PR.pair_counts(pairs)
    for key, row in table.iterrows():
        io.log(f"   {key}: " + ", ".join(f"{k} {v:,}" for k, v in row.items()))
    return pairs, dropped, table


#%% b. Statistics
def step_stats(x: dict, pairs):
    l, e = x["links"], x["events"]
    ev = (e["dep"].to_numpy(), e["ride"].to_numpy(), x["masks"],
          l["ev0"].to_numpy(), l["ev1"].to_numpy(), l["tz_shift"].to_numpy())
    t0 = time.time()
    inst = T.build_instances(*ev)
    typ = T.build_typical(*ev)
    cal = T.link_calendars(np.ascontiguousarray(x["masks"]), ev[3].astype(np.int32),
                           ev[4].astype(np.int32))
    t_build = time.time() - t0
    io.log(f"b. instances: {len(inst.dep):,} dated, {len(typ.dep):,} typical "
           f"({(T.instance_bytes(len(inst.dep), len(l)) + T.instance_bytes(len(typ.dep), len(l))) / 2 ** 20:,.0f} MB); "
           f"{t_build:.1f} s")
    t1 = time.time()
    stats = T.transfer_table(
        inst, typ, cal, e["trip"].to_numpy(), pairs["link_in"].to_numpy(),
        pairs["link_out"].to_numpy(), pairs["move"].to_numpy(),
        pairs["lo"].to_numpy(), pairs["hi"].to_numpy(),
        pairs["typical"].to_numpy())
    t_pairs = time.time() - t1
    io.log(f"b. transfer statistics of {len(pairs):,} pairs in {t_pairs:.1f} s "
           f"({len(pairs) / max(t_pairs, 1e-9) / 1e6:.2f} M pairs/s)")
    return stats, {"build_s": round(t_build, 1), "pairs_s": round(t_pairs, 1)}


#%% Run
def main() -> None:
    t0 = time.time()
    x = read_inputs()
    clock = {"read_s": round(time.time() - t0, 1)}

    t = time.time()
    pairs, dropped, table = step_pairs(x)
    clock["pairs_build_s"] = round(time.time() - t, 1)

    stats, sec = step_stats(x, pairs)
    clock |= sec

    t = time.time()
    transfers = PR.transfers_table(pairs, stats, T.basis_labels)
    ok = PR.exists(transfers, P["MIN_ACTIVE_DATES"])
    by_basis = transfers.groupby("basis").size()
    ok_basis = transfers[ok].groupby("basis").size()
    io.log(f"b. with an opportunity {len(transfers):,} of {len(pairs):,} pairs ("
           + ", ".join(f"{k} {v:,}" for k, v in by_basis.items())
           + "); existing under the consumer's rule "
           + ", ".join(f"{k} {v:,}" for k, v in ok_basis.items()))
    del stats

    ftrans = PR.ftrans_table(transfers, x["links"], x["hubs"], x["members"])
    pooled = int(ftrans["n_rows"].sum())
    clock["tables_s"] = round(time.time() - t, 1)
    io.log(f"c. ps-ftrans: {len(ftrans):,} keys from {pooled:,} of "
           f"{len(transfers):,} rows ({len(transfers) - pooled:,} left out: a hub "
           f"without a released id); {clock['tables_s']:.1f} s")

    io.save(transfers, "ps-transfers")
    io.save(ftrans, "ps-ftrans")
    clock["total_s"] = round(time.time() - t0, 1)
    io.write_manifest("transfers", [io.out_path(n) for n in INPUTS], {
        "pairs": {"admissible": len(pairs), "dropped": dropped,
                  "by_window_and_kind": {f"{k}/{c}": int(v)
                                         for k, row in table.iterrows()
                                         for c, v in row.items()}},
        "transfers": {"rows": len(transfers),
                      "by_basis": {k: int(v) for k, v in by_basis.items()},
                      "existing_by_basis": {k: int(v) for k, v in ok_basis.items()}},
        "ftrans": {"rows": len(ftrans), "transfers_pooled": pooled,
                   "transfers_left_out": len(transfers) - pooled},
        "seconds": clock,
    })
    io.log(f"done in {clock['total_s']:.0f} s")


if __name__ == "__main__":
    main()
