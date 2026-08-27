"""Compute road-network travel-time matrices using OSRM-backend."""

#%%
from itertools import product
import logging
import os
from pathlib import Path
import requests
import shlex
from shutil import rmtree
import subprocess
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

logging.getLogger("pyogrio").setLevel(logging.WARNING)


# Coordinate Reference Systems (CRS)
CRS_DEG = "EPSG:4326" # geographical CRS (unit: degrees)
CRS_M = "EPSG:3035" # spatial CRS (unit: metres)


def run(
    cmd: str | list[str],
    *,
    check: bool = True,
    quiet: bool = False,
    **kwargs: Any
) -> subprocess.CompletedProcess:
    """Run a command as a subprocess."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    if quiet:
        kwargs |= {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL
        }
    return subprocess.run(cmd, check=check, **kwargs)


def _start_server(
    probe_pt: tuple[float, float],
    container: str,
    workdir: Path,
    osrm_img: str,
    profile: str,
    threads: int,
    port: int,
    max_table_size: int,
    start_timeout: float,
    verbosity: str = "ERROR"
) -> subprocess.Popen:
    """Preprocess an OSM extract, start OSRM and await requests."""
    run(f"docker container rm --force {container}", quiet=True, check=False)
    verb = f"--verbosity {verbosity}"
    steps = (
        f"osrm-extract {verb} -p /opt/{profile}.lua "
        f"--threads {threads} /data/tmp.osm.pbf && "
        f"osrm-partition {verb} /data/tmp.osrm && "
        f"osrm-customize {verb} /data/tmp.osrm && "
        f"exec osrm-routed {verb} --algorithm mld --port {port} "
        f"--max-table-size {max_table_size} /data/tmp.osrm"
    )
    process = subprocess.Popen([
        "docker", "run", "--rm", "--volume", f"{workdir}:/data",
        "--name", container, "--publish", f"127.0.0.1:{port}:{port}",
        osrm_img, "sh", "-c", steps
    ], text=True)
    probe_url = (f"http://127.0.0.1:{port}/nearest/v1/driving/" +
                 "{:.6f},{:.6f}".format(*probe_pt))
    deadline = time.monotonic() + start_timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("OSRM server exited with code {}"
                                   .format(process.returncode))
            try:
                requests.get(probe_url, timeout=2)
                return process
            except requests.ConnectionError:
                time.sleep(0.5)
        raise TimeoutError(f"OSRM server did not start in {start_timeout}s")
    except BaseException:
        _stop_server(process, container)
        raise


def _stop_server(process: subprocess.Popen, container: str) -> None:
    """Stop the OSRM Docker container and its attached process."""
    run(f"docker container stop --time 10 {container}", quiet=True, check=False)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _get_table(
    src: pd.DataFrame,
    trg: pd.DataFrame,
    profile: str,
    port: int,
    max_table_size: int
) -> pd.DataFrame:
    """Request a full source-to-target matrix in bounded OSRM batches."""
    mode = {
        "car": "driving",
        "foot": "walking",
        "bicycle": "bicycling"
    }[profile]
    batch_size = max_table_size // 2
    src_batches = [src.iloc[i:i + batch_size]
                   for i in range(0, len(src), batch_size)]
    trg_batches = [trg.iloc[i:i + batch_size]
                   for i in range(0, len(trg), batch_size)]
    tables = []
    for src_batch, trg_batch in product(src_batches, trg_batches):
        pts = pd.concat([src_batch, trg_batch], ignore_index=True)
        n_src = len(src_batch)
        url = f"http://127.0.0.1:{port}/table/v1/{mode}/" + ";".join(
            f"{x:.6f},{y:.6f}" for x, y in zip(pts["x"], pts["y"])
        )
        response = requests.get(url, timeout=300, params={
            "annotations": "distance,duration",
            "sources": ";".join(map(str, range(n_src))),
            "destinations": ";".join(map(
                str, range(n_src, n_src + len(trg_batch))
            ))
        })
        response.raise_for_status()
        data = response.json()
        if data["code"] != "Ok":
            raise RuntimeError(data.get("message", data["code"]))

        tables.append(pd.DataFrame({
            "src_id": np.repeat(src_batch.index, len(trg_batch)),
            "trg_id": np.tile(trg_batch.index, len(src_batch)),
            "time": np.asarray(data["durations"], dtype=float).ravel(),
            "dist": np.asarray(data["distances"], dtype=float).ravel()
        }))
    df = pd.concat(tables, ignore_index=True)
    df = df.astype({"src_id": np.int32, "trg_id": np.int32,
                    "time": np.float32, "dist": np.float32})
    return df


def get_travel_times(
    src: pd.DataFrame,
    trg: pd.DataFrame,
    osm_path: str | Path,
    buffer_radius: float = 20,
    profile: str = "car",
    port: int = 5108,
    threads: int = max(1, min(4, (os.cpu_count() or 2) // 2)),
    max_table_size: int = 500,
    server_start_timeout: float = 180,
    workdir: str | Path = ".tmp",
    osrm_img: str = "ghcr.io/project-osrm/osrm-backend:v5.27.1"
) -> pd.DataFrame:
    """Compute a full road-network matrix between two point tables.

    Parameters
    ----------
    src, trg : pd.DataFrame
        Source and target points with `x` (longitude) and `y` (latitude)
        columns in WGS 84.
    osm_path : str | Path
        OSM PBF extract covering the source and target points. For repeated
        FUAs, this can be a prefiltered country-union extract.
    buffer_radius : float, default 20
        Buffer in kilometres around the convex hull of all input points.
    profile : {"car", "foot", "bicycle"}, default "car"
        OSRM routing profile.
    port : int, default 5108
        Local port exposed by the temporary OSRM container.
    threads : int
        Threads used by `osrm-extract`.
    max_table_size : int, default 500
        Maximum coordinates accepted by one OSRM table request.
    server_start_timeout : float, default 180
        Maximum seconds for preprocessing and `osrm-routed` startup.
    workdir : str | Path
        Temporary directory for the clipped PBF and OSRM files. The directory
        is deleted before and after every call.
    osrm_img : str
        Versioned OSRM-backend Docker image.

    Returns
    -------
    pd.DataFrame
        Long-form matrix with `src_id`, `trg_id`, `time` (seconds) and `dist`
        (metres). IDs are zero-based row positions in `src` and `trg`.
    """
    assert {"x", "y"}.issubset(src.columns)
    assert {"x", "y"}.issubset(trg.columns)
    assert len(src) > 0 and len(trg) > 0
    assert max_table_size >= 2
    
    ## Resolve origin & destination points
    src = src[["x", "y"]].reset_index(drop=True).rename_axis("src_id")
    trg = trg[["x", "y"]].reset_index(drop=True).rename_axis("trg_id")
    pts = pd.concat([src, trg], ignore_index=True)
    geom = gpd.points_from_xy(pts["x"], pts["y"], crs=CRS_DEG)
    pts = gpd.GeoDataFrame(geometry=geom).rename_axis("pt_id")
    
    ## Configure
    container = "osrm-times"
    workdir = Path(workdir).resolve()
    rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        ## Clip the supplied OSM extract to the OD extent
        extent = gpd.GeoDataFrame(geometry=[
            pts.to_crs(CRS_M).union_all().convex_hull
            .buffer(buffer_radius * 1000)
        ], crs=CRS_M)
        json_path = workdir / "boundary.geojson"
        extent.to_crs(CRS_DEG).to_file(json_path, driver="GeoJSON")
        run(
            "osmium extract --strategy=complete_ways --overwrite "
            f"-p {json_path} -o {workdir}/tmp.osm.pbf {osm_path}"
        )
        ## Preprocess the extract and start OSRM server
        probe_pt = tuple(pts.geometry.iloc[0].coords[0])
        server = _start_server(
            probe_pt, container, workdir, osrm_img, profile, threads, 
            port, max_table_size, server_start_timeout
        )
        ## Request travel time matrix
        ttm = _get_table(src, trg, profile, port, max_table_size)
        if ttm is not None:
            ttm = ttm.loc[
                np.isfinite(ttm["dist"]) & np.isfinite(ttm["time"]) &
                ~np.isnan(ttm["dist"]) & ~np.isnan(ttm["time"])
            ].query("src_id != trg_id").reset_index(drop=True)
        return ttm
    except Exception as e:
        print("ERROR:", e)
    finally:
        if server is not None:
            _stop_server(server, container)
        rmtree(workdir, ignore_errors=True)
        
# x = get_travel_times(pts, pts, osm_path, workdir=C.DATA / "tmp"); x

#%% Test run
if __name__ == "__main__":
    import config as C

    fuas = C.load("fuas")
    stns = C.load("ic-stations")

    city = "Amsterdam"
    fua = fuas[fuas["name"] == city].iloc[0]
    icc = fua["icc"]
    osm_path = C.DATA / f"osm/citygroup/{icc}.osm.pbf"
    pts = stns[stns.geometry.within(fua.geometry)].get_coordinates()
    ttm = get_travel_times(pts, pts, osm_path, workdir=C.DATA / "tmp")
