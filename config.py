"""Custom configuration and utility tools for this project."""

#%% Imports
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import geopandas as gpd
from IPython.display import display
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

#%% Constants
# Geographical coordinate reference system (CRS) of WGS-84 (unit: degrees)
CRS_DEG = "EPSG:4326"
# Spatial CRS best suited for Europe (unit: metres)
CRS_EU = "EPSG:3035"

#%% User-specific environment variables
try:
    with open("env.yml", "r") as f:
        env_data = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Environment file `./env.yml` not found.")

# Main data directory for the project
DATA = Path(env_data.get("DATA_DIR", "./data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)

# Folder for output figures [optional]
FIG = Path(env_data.get("FIG_DIR", "./fig")).resolve()
FIG.mkdir(parents=True, exist_ok=True)

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

def _map(df: gpd.GeoDataFrame, *args, tiles="CartoDB.Voyager", **kwargs):
    return df.explore(*args, tiles=tiles, **kwargs)
    
gpd.GeoDataFrame.map = _map


#%% Plotting
pyplot_params = {
    "axes.edgecolor": "k",
    "axes.edgecolor": "k",
    "axes.formatter.use_mathtext": True,
    "axes.grid": True,
    "axes.labelcolor": "k",
    "axes.labelsize": 13,
    "axes.linewidth": 0.5,
    "axes.titlesize": 15,
    "figure.dpi": 150,
    "figure.titlesize": 15,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Noto Sans", "DejaVu Serif"],
    "grid.alpha": 0.15,
    "grid.color": "k",
    "grid.linewidth": 0.5,
    "legend.edgecolor": "none",
    "legend.facecolor": ".9",
    "legend.fontsize": 11,
    "legend.framealpha": 0.5,
    "legend.labelcolor": "k",
    "legend.title_fontsize": 13,
    "mathtext.fontset": "cm",
    "text.color": "k",
    "text.color": "k",
    "xtick.bottom": True,
    "xtick.color": "k",
    "xtick.labelsize": 10,
    "xtick.minor.visible": True,
    "ytick.color": "k",
    "ytick.labelsize": 10,
    "ytick.left": True,
    "ytick.minor.visible": True,
}

def imsave(title=None, fig=None, ax=None, dpi=300,
           root=FIG, ext="png", opaque=True):
    """Save the current matplotlib figure to disk."""
    fig = fig or plt.gcf()
    ax = ax or fig.axes[0]
    time = datetime.now().strftime("%Y-%m-%d_%H-%m-%S")
    title = title or fig._suptitle or ax.get_title() or f"Untitled {time}"
    fig.savefig(
        f"{mkdir(root)}/{title}.{ext}",
        dpi=dpi,
        bbox_inches="tight",
        transparent=not opaque,
        facecolor="white" if opaque else "auto",
    )
