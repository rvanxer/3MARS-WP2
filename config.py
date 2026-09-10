"""Custom configuration and utility tools for this project."""

#%% Imports
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import geopandas as gpd
from IPython.display import display
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

#%% Constants
# Geographical coordinate reference system (CRS) of WGS-84 (unit: degrees)
CRS_DEG = "EPSG:4326"
# Spatial CRS best suited for Europe (unit: metres)
CRS_EU = "EPSG:3035"


#%% URLs/URIs of data sources
with open("urls.yml", "r") as f:
    URLS = yaml.safe_load(f)


#%% User-specific environment variables
with open("env.yml", "r") as f:
    env_data = yaml.safe_load(f)

# Main data directory for the project
DATA = Path(env_data.get("DATA_DIR", "./data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)

# MobilityDatabase API key
MDB_API_KEY = env_data.get("MDB_API_KEY")

# CartoDB access token for plotting basemaps
CARTO_TOKEN = env_data.get("CARTO_TOKEN")


#%% Logging
def setup_logger(
    name: str = "",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers if setup_logger() is called more than once
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logger()

def log(msg: str, logger=logger):
    logger.info(msg)
    
def warn(msg: str, logger=logger):
    logger.warning(msg)

def error(msg: str, logger=logger):
    logger.error(msg)


#%% File handling
def mkdir(path):
    """Shorthand for making a folder if it does not exist."""
    assert isinstance(path, str) or isinstance(path, Path)
    Path(path).mkdir(exist_ok=True, parents=True)
    return Path(path)


def load(name: str, root=DATA, quiet=False, **kwargs):
    """Load a processed parquet file from a given folder into a dataframe."""
    path = Path(root) / f"{name}.parquet"
    if path.exists():
        if "geometry" in pq.ParquetFile(path).schema.names:
            if "columns" in kwargs and "geometry" not in kwargs["columns"]:
                df = pd.read_parquet(path, **kwargs)
            df = gpd.read_parquet(path, **kwargs)
        else:
            df = pd.read_parquet(path, **kwargs)
        if not quiet:
            log(f"Loaded table '{name}'")
        return df
    if not quiet:
        error(f"Table {name} not found")


def save(df: pd.DataFrame | gpd.GeoDataFrame,
         name: str, root=DATA, **kwargs):
    """Write a processed dataframe to a parquet file in a given folder."""
    df.to_parquet(root / f"{name}.parquet", **kwargs)
    log(f"Saved table '{name}'")
    

def load_params(yml_file: str | Path = Path("params.yml")):
    """Load the study parameters into a simple namespace object."""
    with open(yml_file, "r") as f:
        params = yaml.safe_load(f)
    log("Loaded parameters")
    return SimpleNamespace(**params)


#%% Data handling
def pdf2gdf(df: pd.DataFrame, x: str = "lon", y: str = "lat",
            crs=None, drop_cols=True) -> gpd.GeoDataFrame:
    """Convert a pandas DataFrame to a geopandas GeoDataFrame by creating
    point geometry from the dataframes x & y columns."""
    geom = gpd.points_from_xy(df[x], df[y], crs=crs)
    df = gpd.GeoDataFrame(df, geometry=geom)
    if drop_cols:
        df.drop(columns=[x, y], errors="ignore", inplace=True)
    return df


def normalise(x: Union[np.array, pd.Series], vmin=None, vmax=None):
    """Range normalise a vector."""
    vmin = vmin or np.min(x)
    vmax = vmax or np.max(x)
    return (x - vmin) / (vmax - vmin)


def _view_pdf(df: pd.DataFrame, nrow: int = 0):
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    log("{:,} rows x {:,} cols [{:.1f} MiB]".format(*df.shape, mem))
    display(pd.concat([pd.DataFrame({
        df.index.name or "": "(" + df.dtypes.astype(str) + ")"
    }).T, df.head(nrow)]))
    return df

def _view_gdf(df: gpd.GeoDataFrame, nrow: int = 0):
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    log("{:,} rows x {:,} cols [{:.1f} MiB] [CRS: EPSG:{}]".format(
        *df.shape, mem, df.crs.to_epsg()))
    display(pd.concat([pd.DataFrame({
        df.index.name or "": "(" + df.dtypes.astype(str) + ")"
    }).T, df.head(nrow).astype({"geometry": str})]))
    return df

# add the `disp` method to pandas and geopandas series & DF classes
pd.DataFrame.view = _view_pdf
gpd.GeoDataFrame.view = _view_gdf
