"""Car access and egress, and moves between hubs (plan step 2).

Reads ps-hubs (pathset-network.py), the released graph (RELEASE_DIR
nodes.csv, links.csv), data-upstream fuas and airports, data-upstream-hubs
air2rail. Writes ps-connectors and ps-moves (contract: pathset/schemas.py).

Run:  python -X utf8 pathset-connectors.py
"""

#%% Imports
import time

import pandas as pd

from pathset import connect as C
from pathset import io

P = io.P


#%% Inputs
def read_inputs() -> dict:
    nodes = pd.read_csv(io.RELEASE / "nodes.csv",
                        usecols=["node_id", "lon", "lat"])
    links = pd.read_csv(io.RELEASE / "links.csv",
                        usecols=["src", "trg", "kind", "mode", "time"])
    return {
        "hubs": io.load_out("ps-hubs"),
        "rel": C.released_tables(nodes, links),
        "centres": C.fua_centres(io.load("fuas", io.DATA, ["id", "centre"])),
        "airports": io.load("airports", io.DATA, ["iata", "fua"]),
        "air2rail": io.load("air2rail", io.HUBS, ["iata", "stn", "tier"]),
    }


def fit_text(name: str, f: C.Fit) -> str:
    return (f"fit {name}: minutes = {f.a:.2f} + {f.b:.3f} x km; R2 {f.r2:.3f}, "
            f"MAE {f.mae:.1f}, leave-one-out MAE {f.loo_mae:.1f} min, n {f.n}")


#%% Run
def main() -> None:
    t0 = time.time()
    x = read_inputs()
    io.log(f"inputs read in {time.time() - t0:.1f} s")

    fits = C.released_fits(x["rel"], x["centres"])
    for name, f in fits.items():
        io.log(fit_text(name, f))

    t1 = time.time()
    conn = C.build_connectors(x["hubs"], x["airports"], x["rel"], x["centres"],
                              fits, P["ACCESS_MAX_AIR_MIN"])
    kind = conn["hub"].map(x["hubs"].set_index("hub")["kind"])
    by = conn.groupby([kind, conn["source"]]).size()
    io.log("connectors by hub kind and source: " + ", ".join(
        f"{k}/{s} {n:,}" for (k, s), n in by.items())
        + f"; {conn['fua'].nunique()} FUAs; {time.time() - t1:.1f} s")

    t2 = time.time()
    moves = C.build_moves(x["hubs"], x["air2rail"], x["rel"], fits, P)
    io.log("moves by kind: " + ", ".join(
        f"{k} {n:,}" for k, n in moves["kind"].value_counts().items())
        + f"; {time.time() - t2:.1f} s")

    io.save(conn, "ps-connectors")
    io.save(moves, "ps-moves")
    io.write_manifest(
        "connectors",
        [io.out_path("ps-hubs.parquet"), io.RELEASE / "nodes.csv",
         io.RELEASE / "links.csv", io.DATA / "fuas.parquet",
         io.DATA / "airports.parquet", io.HUBS / "air2rail.parquet"],
        {"fits": {k: f._asdict() for k, f in fits.items()},
         "connectors": {f"{k}/{s}": int(n) for (k, s), n in by.items()},
         "moves": {k: int(n) for k, n in moves["kind"].value_counts().items()},
         "seconds": round(time.time() - t0, 1)})
    io.log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
