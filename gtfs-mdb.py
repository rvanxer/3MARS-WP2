"""Download GTFS feeds closest to a fixed MobilityDatabase snapshot date."""

#%% Imports
from datetime import datetime, time, timezone
from hashlib import sha256
from time import sleep

from mobility_db_api import MobilityAPI
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import config as C

params = C.load_params()

#%% Countries and snapshot date of interest
countries = C.load("countries")#.view()
snapshot_date = params.MDB_SNAPSHOT_DATE
snapshot_end = datetime.combine(snapshot_date, time.max, timezone.utc)
snapshot_str = snapshot_date.strftime("%y%m%d")
outdir = C.mkdir(C.DATA / f"gtfs/feeds")

#%% MobilityDatabase API [4s]
if not C.MDB_API_KEY:
    raise RuntimeError("Mobility Database refresh token not found. "
                       "Set MDB_API_KEY in `env.yml`.")
mdb_api = MobilityAPI(refresh_token=C.MDB_API_KEY)
session = requests.Session()
headers = {"Authorization": f"Bearer {mdb_api.get_access_token()}"}


def get_datasets(feed: str, **params) -> list[dict]:
    """Get one page of archived datasets for an MDB feed."""
    url = f"{mdb_api.base_url}/gtfs_feeds/{feed}/datasets"
    response = session.get(url, headers=headers, params=params, timeout=60)
    if response.status_code == 401:
        headers["Authorization"] = f"Bearer {mdb_api.get_access_token()}"
        response = session.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def get_closest_dataset(feed: str, limit=500) -> tuple[dict | None, str]:
    """Select the newest dataset on/before the snapshot, else the first after."""
    end = snapshot_end.isoformat()
    if (before := get_datasets(feed, limit=1, downloaded_before=end)):
        return before[0]
    after = []
    n = 0 # offset with the batch
    while True:
        batch = get_datasets(feed, limit=limit, downloaded_after=end, offset=n)
        after.extend(batch)
        if len(batch) < limit:
            break
        n += limit
    if after:
        return min(after, key=lambda x: x["downloaded_at"])


def get_snapshot_dataset(feed: dict) -> tuple[dict | None, str]:
    """Use the latest dataset unless it is newer than the snapshot."""
    latest = feed["latest_dataset"]
    if latest and (date := latest.get("downloaded_at")):
        if pd.to_datetime(date, utc=True) <= snapshot_end:
            return latest
    return get_closest_dataset(feed["id"])


def file_hash(path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 ** 2):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(feed: str, url: str, expected_hash: str,
                     outdir=outdir) -> str:
    """Download and verify the selected MDB dataset."""
    outfile = outdir / f"{feed}.zip"
    if outfile.exists() and file_hash(outfile) == expected_hash:
        return expected_hash

    tempfile = outfile.with_suffix(".zip.part")
    response = session.get(url, stream=True, timeout=300)
    response.raise_for_status()
    digest = sha256()
    with open(tempfile, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 ** 2):
            if chunk:
                f.write(chunk)
                digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        tempfile.unlink()
        return download_dataset(feed, url, expected_hash=actual_hash)
    tempfile.replace(outfile)
    return actual_hash

#%% MDB dataset catalog [4m41s]
catalog_file = C.DATA / f"gtfs/mdb-feeds-{snapshot_str}.csv"
if not catalog_file.exists():
    C.log(f"Identifying MDB datasets closest to {snapshot_date}")
    feeds = []
    for icc in (pbar := tqdm(countries.index)):
        pbar.set_description(icc)
        for feed in mdb_api.get_providers_by_country(icc):
            try:
                if (
                    feed["data_type"] == "gtfs" and
                    feed["status"] != "deprecated" and
                    (ds := get_snapshot_dataset(feed))
                ):
                    date = pd.to_datetime(ds["downloaded_at"], utc=True)
                    feeds.append(dict(
                        name = feed["id"],
                        provider = feed["provider"],
                        status = feed["status"].title(),
                        download_date = str(date.date()),
                        delta_days = (date.date() - snapshot_date).days,
                        url = ds["hosted_url"],
                        hash = ds["hash"],
                        start_date = ds.get("service_date_range_start"),
                        end_date = ds.get("service_date_range_end"),
                    ))
            except Exception as e:
                C.error(f"{feed['id']}: {e}")
        sleep(2)
        pbar.update()
    feeds = (
        pd.DataFrame(feeds)
        .drop_duplicates("name")
        .sort_values("name", ignore_index=True)
    )
    feeds.to_csv(catalog_file, index=False)
feeds = pd.read_csv(catalog_file)#.view()

#%% Download and verify the selected MDB feeds [10m40s]
# Remove existing feeds not in the current MDB catalog
for file in sorted(outdir.glob("*.zip")):
    feed = file.stem
    # skip the manual feeds (that start with "man-")
    if feed not in list(feeds["name"]) and not feed.startswith("man-"):
        file.unlink()
        C.warn(f"Deleted stale feed '{feed}'")

C.log(f"Downloading MDB snapshot {snapshot_date}")
if "local_hash" not in feeds.columns:
    feeds["local_hash"] = None
for i, row in (pbar := tqdm(feeds.iterrows(), total=len(feeds))):
    pbar.set_description(row["name"])
    try:
        hash_ = download_dataset(row["name"], row["url"], row["hash"])
        feeds.loc[i, "local_hash"] = hash_
        pbar.update()
    except Exception as e:
        C.error(f"{row['name']}: {e}")
