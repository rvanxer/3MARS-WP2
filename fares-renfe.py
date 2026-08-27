#%% Imports
import kagglehub
import numba as np
import pandas as pd

import config as C

params = C.load_params()

#%% RENFE fares data [35s]
fname = "thegurus-opendata-renfe-trips.csv"
try:
    csv_path = kagglehub.dataset_download(
        "thegurusteam/spanish-high-speed-rail-system-ticket-pricing",
        fname, output_dir=C.DATA / "price"
    )
except FileExistsError:
    csv_path = C.DATA / "price/data" / fname
fares = (
    pd.read_csv(csv_path, usecols=list(cols := {
        "origin": "category",
        "destination": "category",
        "departure": "category",
        "arrival": "category",
        "vehicle_class": "category",
        "vehicle_type": "category",
        "fare": "category",
        "price": "Float32",
        "seats": "Float32",
        "meta": "category",
    })).dropna(subset="price", ignore_index=True)
    .astype(cols)
).view()
