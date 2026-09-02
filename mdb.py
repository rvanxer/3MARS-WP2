"""Download a fixed-date GTFS snapshot from MobilityDatabase."""

#%% Imports
import datetime as dt
from hashlib import sha256

import pandas as pd
import requests
from tqdm import tqdm

import config as C


#%% Configuration
params = C.load_params()
SNAPSHOT_DATE = dt.datetime.combine(params.MDB_SNAPSHOT_DATE, dt.time())

COUNTRIES = sorted(
    C.load("countries")["icc"]
    .replace({"UK": "GB", "EL": "GR"})
    .unique()
)
FEED_DIR = C.DATA / "gtfs/feeds"

#%% MobilityDatabase API
class API:
    """Small replacement for mobility_db_api.MobilityAPI."""

    base_url = "https://api.mobilitydatabase.org/v1"
    token_error = ("Mobility Database refresh token not found. "
                   "Set MDB_API_KEY in `env.yml`.")

    def __init__(
        self,
        refresh_token: str,
        feed_page_size: int = 2500,
        dataset_page_size: int = 500,
    ):
        if not refresh_token:
            raise RuntimeError(self.token_error)

        self.feed_page_size = feed_page_size
        self.dataset_page_size = dataset_page_size
        self.session = requests.Session()
        response = self.session.post(
            f"{self.base_url}/tokens",
            json={"refresh_token": refresh_token},
            timeout=60)
        if response.status_code in (400, 401, 403):
            raise RuntimeError(self.token_error)
        response.raise_for_status()

        access_token = response.json().get("access_token")
        if not access_token:
            raise RuntimeError(self.token_error)
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def _get(self, path: str, **params):
        response = self.session.get(
            f"{self.base_url}{path}", headers=self.headers,
            params=params, timeout=120)
        if response.status_code == 401:
            raise RuntimeError(self.token_error)
        response.raise_for_status()
        return response.json()

    def get_feeds_by_country(self, icc: str) -> list[dict]:
        """Get every page of GTFS feeds for a country."""
        feeds = []
        offset = 0
        while True:
            batch = self._get(
                "/gtfs_feeds", country_code=icc,
                limit=self.feed_page_size, offset=offset)
            if not isinstance(batch, list):
                raise TypeError(f"{icc}: feeds response is not a list")
            feeds.extend(batch)
            if len(batch) < self.feed_page_size:
                return feeds
            offset += self.feed_page_size

    def get_datasets(self, feed: str, **params) -> list[dict]:
        """Get one page of archived datasets for a GTFS feed."""
        params.setdefault("limit", self.dataset_page_size)
        datasets = self._get(f"/gtfs_feeds/{feed}/datasets", **params)
        if not isinstance(datasets, list):
            raise TypeError(f"{feed}: datasets response is not a list")
        return datasets


#%% Catalog
def build_catalog(
    snapshot_date: dt.datetime = SNAPSHOT_DATE,
    country_codes: list[str] = COUNTRIES,
    rebuild=False,
) -> pd.DataFrame:
    """Build the MobilityDatabase catalogue for a fixed snapshot date."""
    snapshot_end = dt.datetime.combine(
        snapshot_date, dt.time.max, dt.timezone.utc)
    snapshot_str = snapshot_date.strftime("%y%m%d")
    catalog_file = C.DATA / f"gtfs/mdb-catalog-{snapshot_str}.csv"

    if catalog_file.exists() and not rebuild:
        return pd.read_csv(catalog_file)

    api = API(
        refresh_token=C.MDB_API_KEY,
        feed_page_size=2500,
        dataset_page_size=500)

    def get_snapshot_dataset(feed: dict) -> dict | None:
        latest = feed.get("latest_dataset")
        if latest and (date := latest.get("downloaded_at")):
            if pd.to_datetime(date, utc=True) <= snapshot_end:
                return latest
        try:
            datasets = api.get_datasets(
                feed["id"], limit=1,
                downloaded_before=snapshot_end.isoformat())
        except TypeError:
            return None
        if datasets:
            return datasets[0]

    C.log(f"Identifying MDB datasets closest to {snapshot_date.date()}")
    feeds = []
    for country in (pbar := tqdm(sorted(country_codes))):
        pbar.set_description(country)
        for feed in api.get_feeds_by_country(country):
            if (
                feed["data_type"] == "gtfs" and
                (dataset := get_snapshot_dataset(feed))
            ):
                date = pd.to_datetime(dataset["downloaded_at"], utc=True)
                delta = (date.date() - snapshot_date.date()).days
                feeds.append(dict(
                    name = feed["id"],
                    provider = feed["provider"],
                    status = feed["status"],
                    download_date = str(date.date()),
                    delta_days = delta,
                    url = dataset["hosted_url"],
                    hash = dataset["hash"],
                    start_date = dataset.get("service_date_range_start"),
                    end_date = dataset.get("service_date_range_end"),
                ))
    catalog = (
        pd.DataFrame(feeds)
        .drop_duplicates("name")
        .sort_values("name", ignore_index=True)
    )
    catalog.to_csv(catalog_file, index=False)
    C.log(f"Saved {len(catalog):,} feeds to '{catalog_file}'")
    return catalog

catalog = build_catalog(rebuild=False).view() # 4m5s

#%% Download feeds
def download_feeds(
    catalog: pd.DataFrame,
    outdir=FEED_DIR,
    overwrite=False,
    remove_stale=True,
) -> None:
    """Download and verify every feed, preserving manual GTFS archives."""
    C.log(f"Downloading MDB feeds")
    outdir = C.mkdir(outdir)
    n_download = 0
    for _, row in (pbar := tqdm(catalog.iterrows(), total=len(catalog))):
        pbar.set_description(feed_id := row["name"])
        # check the file hashes
        hash_ = row["hash"] # expected hash from the MDB catalog
        hash_ = hash_ if hash_ and not pd.isna(hash_) else None
        digest = sha256() # identify the hash of the current data file
        outfile = outdir / f"{feed_id}.zip"
        if outfile.exists() and hash_:
            with open(outfile, "rb") as f:
                while chunk := f.read(1024 ** 2):
                    digest.update(chunk)
            file_hash = digest.hexdigest()
            if file_hash == hash_ and not overwrite:
                # C.log(f"Skipping existing {feed_id}")
                continue
        tempfile = outfile.with_suffix(".zip.part")
        digest = sha256()
        try:
            with requests.get(row["url"], stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(tempfile, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 ** 2):
                        if chunk:
                            f.write(chunk)
                            digest.update(chunk)
            actual_hash = digest.hexdigest()
            if hash_ and actual_hash != hash_:
                C.error(f"{feed_id}: Hash mismatch: {hash_}; {actual_hash}")
                continue
            tempfile.replace(outfile)
            n_download += 1
        finally:
            tempfile.unlink(missing_ok=True)
    if remove_stale:
        removed = []
        for file in sorted(outdir.glob("*.zip")):
            manually_downloaded = file.stem.startswith("man-")
            in_catalog = file.stem in set(catalog["name"])
            if not in_catalog and not manually_downloaded:
                file.unlink()
                removed.append(file.stem)
        C.warn(f"Deleted {len(removed)} stale feeds: " + ", ".join(removed))
    C.log(f"Downloaded and verified {n_download:,} MDB feeds")

x = download_feeds(catalog, overwrite=False, remove_stale=True); x
