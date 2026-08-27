"""Download latest GTFS feeds from the MobilityDatabase (MDB)."""

#%% Imports
from shutil import rmtree
from time import sleep

from mobility_db_api import MobilityAPI
import numpy as np
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile, ZIP_DEFLATED

import config as C

#%% Countries of interest
countries = C.load("countries").view()

#%% MobilityDatabase feed information (inc. download URL)
if not C.MDB_API_KEY:
    raise RuntimeError(
        "Mobility Database refresh token not found. Create the untracked "
        "mdb-key-api.txt file or set MDB_API_KEY_FILE in env.yml."
    )
mdb_api = MobilityAPI(refresh_token=C.MDB_API_KEY)
feeds = C.load("gtfs/mdb-feed-info")
if feeds is None:
    C.log("Obtaining MDB provider information")
    feeds = []
    pbar = tqdm(total=len(countries))
    for batch in np.array_split(countries.index, 4):
        for i in batch:
            icc = countries.icc.iloc[i]
            pbar.set_description(f"{i+1} {icc}")
            for f in mdb_api.get_providers_by_country(icc):
                if f["data_type"] != "gtfs" or f["status"] == "deprecated":
                    continue
                if (ds := f["latest_dataset"]):
                    if len(date := ds["id"].split("-")[-1]) == 1:
                        date = ds["id"].split("-")[-2]
                    date = str(pd.to_datetime(date[:8], format="%Y%m%d").date())
                    feeds.append(dict(
                        icc = icc,
                        name = f["id"],
                        provider = f["provider"],
                        status = f["status"].title(),
                        date = date,
                        url = ds["hosted_url"]
                    ))
        sleep(2)
    feeds = (pd.DataFrame(feeds)
            .drop_duplicates("name")
            .sort_values("name", ignore_index=True)).view()
    C.save(feeds, "gtfs/mdb-feed-info")

#%% Download original MDB feeds [~1:32:00]
outdir = C.mkdir(C.DATA / "gtfs/feeds")
C.log("Downloading latest MDB feeds")
for feed in (pbar := tqdm(feeds.name)):
    pbar.set_description(feed)
    try:
        outfile = outdir / f"{feed}.zip"
        if outfile.exists():
            continue
        mdb_api.reload_metadata()
        root = mdb_api.download_latest_dataset(feed, download_dir=outdir)
        with ZipFile(outfile, "w", ZIP_DEFLATED) as zipf:
            for f in root.glob("*.txt"):
                zipf.write(f, f.name)
        rmtree(root.parent)
    except Exception as e:
        print(f"ERROR in {feed}: {e}")
