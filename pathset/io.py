"""Folders, parameters and table input/output of the path-set module.

The repository's ``config.py`` is deliberately not used: it reads its YAML
files in the system encoding (cp932 on this machine), imports IPython and
creates the data folder on import. Everything here reads UTF-8 and writes
only inside ``PATHSET_DIR``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "pathset-params.yml"


def load_params(path: Path = PARAMS_FILE) -> dict:
    """Read the parameter file as UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


P = load_params()


def _folder(key: str) -> Path:
    return (ROOT / P[key]).resolve()


HUBS = _folder("HUBS_DIR")
DATA = _folder("DATA_DIR")
RELEASE = _folder("RELEASE_DIR")
OUT = _folder("PATHSET_DIR")


# --- Logging -----------------------------------------------------------------
def _logger() -> logging.Logger:
    logger = logging.getLogger("pathset")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log(msg: str) -> None:
    _logger().info(msg)


# --- Reading -----------------------------------------------------------------
def load(name: str, root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``<root>/<name>.parquet``. Geometry columns stay as WKB bytes."""
    path = Path(root) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def out_path(name: str) -> Path:
    """Path of an output file; refuses anything outside ``PATHSET_DIR``."""
    path = (OUT / name).resolve()
    if OUT not in path.parents:
        raise ValueError(f"{path} is outside {OUT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_out(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return load(name, OUT, columns)


# --- Writing (atomic: temporary name, then rename) ---------------------------
def save(df: pd.DataFrame, name: str) -> Path:
    """Write ``PATHSET_DIR/<name>.parquet`` atomically."""
    path = out_path(f"{name}.parquet")
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log(f"saved {name}: {len(df):,} rows")
    return path


def save_npy(arr: np.ndarray, name: str) -> Path:
    """Write ``PATHSET_DIR/<name>.npy`` atomically."""
    path = out_path(f"{name}.npy")
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    log(f"saved {name}.npy: shape {arr.shape}")
    return path


def load_npy(name: str) -> np.ndarray:
    return np.load(out_path(f"{name}.npy"))


# --- Manifest ----------------------------------------------------------------
def fingerprint(paths: list[Path]) -> dict:
    """Size and modification time of input files (cheap change detection)."""
    out = {}
    for p in paths:
        st = Path(p).stat()
        out[str(p)] = [st.st_size, int(st.st_mtime)]
    return out


def write_manifest(step: str, inputs: list[Path], info: dict | None = None) -> None:
    """Append one record to ``ps-manifest.jsonl``."""
    import numba
    import pyarrow

    rec = {
        "step": step,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "numba": numba.__version__},
        "params": P,
        "inputs": fingerprint(inputs),
        "info": info or {},
    }
    with open(out_path("ps-manifest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
