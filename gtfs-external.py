"""Download and prepare GTFS feeds outside of MobilityDatabase."""

#%% Imports
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import requests
from tqdm import tqdm

import config as C


#%% Download utilities
FEEDS_DIR = C.mkdir(C.DATA / "gtfs/feeds")


def download(url: str, outfile: str | Path, expected_hash=None):
    """Download one file atomically and optionally verify its SHA-256 hash."""
    outfile = Path(outfile)
    if outfile.exists():
        return outfile
    tempfile = outfile.with_suffix(outfile.suffix + ".part")
    digest = sha256()
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tempfile, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 ** 2):
                    if chunk:
                        f.write(chunk)
                        digest.update(chunk)
        if expected_hash and digest.hexdigest() != expected_hash:
            raise ValueError(
                f"Hash mismatch for {outfile.name}: "
                f"{expected_hash}; {digest.hexdigest()}"
            )
        tempfile.replace(outfile)
    finally:
        tempfile.unlink(missing_ok=True)
    return outfile


def validate_gtfs(file):
    """Check that an archive contains the core GTFS tables and is readable."""
    with ZipFile(file) as zf:
        broken_file = zf.testzip()
        if broken_file:
            raise ValueError(f"Corrupt ZIP member in {file}: {broken_file}")
        tables = {
            Path(name).stem
            for name in zf.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() == ".txt"
        }
    missing = {"agency", "routes", "stops", "trips", "stop_times"} - tables
    if missing:
        raise ValueError(f"{file} is missing GTFS tables: {sorted(missing)}")


def publish_feed(file, name):
    """Copy a staged manual feed into the directory consumed by gtfs-db.py."""
    validate_gtfs(file)
    outfile = FEEDS_DIR / f"ext-{name}.zip"
    if not outfile.exists():
        shutil.copy2(file, outfile)
        C.log(f"Published '{outfile.name}'")
    else:
        validate_gtfs(outfile)
    return outfile


#%% Bulgarian rail: resolve the current GTFS file from the BGNAP catalogue
def resolve_bdz(source):
    root = source["api-root"]
    resp = requests.get(
        f"{root}/datasets/{source['dataset-id']}/subsets",
        params={"locale": "en"},
        timeout=60,
    )
    resp.raise_for_status()
    subset = next(
        item for item in resp.json()
        if item["format"] == "gtfs-static" and item["is_active"]
    )
    resp = requests.get(
        f"{root}/subsets/{subset['id']}/files",
        params={"format": "gtfs-static"},
        timeout=60,
    )
    resp.raise_for_status()
    file = next(item for item in resp.json() if item["is_latest"])
    return f"{root}/files/{file['id']}/download", file.get("checksum")


def download_bg_rail(source):
    outfile = FEEDS_DIR / "ext-BDZ.zip"
    if not outfile.exists():
        C.log("Downloading Bulgarian rail feed")
        url, expected_hash = resolve_bdz(source)
        download(url, outfile, expected_hash)
    return publish_feed(outfile, "BDZ")


#%% Romanian rail: download official XML and run a pinned converter revision
def romanian_resources(source):
    response = requests.get(
        f"{source['api-root']}/package_search",
        params=dict(
            fq=f"organization:{source['organisation']}",
            rows=100,
        ), timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise RuntimeError("The Romanian CKAN catalogue query failed")
    period = source["timetable-period"]
    resources = []
    for package in result["result"]["results"]:
        matches = [
            resource for resource in package["resources"]
            if resource.get("format", "").lower().lstrip(".") == "xml"
            and period in resource.get("name", "")
        ]
        if matches:
            resources.append(max(matches, key=lambda x: 
                x.get("last_modified") or x.get("created") or ""
            ))
    if not resources:
        raise RuntimeError(f"No Romanian XML resources found for {period}")
    return resources


def download_romania(source):
    outfile = FEEDS_DIR / "ext-Romania.zip"
    if outfile.exists():
        return publish_feed(outfile, "Romania")
    if (FEEDS_DIR / outfile.name).exists():
        return FEEDS_DIR / outfile.name

    ruby = shutil.which("ruby")
    if ruby is None:
        raise RuntimeError("Ruby is required for Romanian GTFS converter")
    nokogiri = subprocess.run(
        [ruby, "-e", 'require "nokogiri"'],
        capture_output=True,
        text=True,
    )
    if nokogiri.returncode:
        raise RuntimeError("Ruby package 'nokogiri' is required: " +
                           "gem install nokogiri")

    with TemporaryDirectory(prefix="gtfs-ro-") as tmpdir:
        tmpdir = Path(tmpdir)
        converter_zip = download(source["converter-url"],
                                 tmpdir / "converter.zip")
        with ZipFile(converter_zip) as zf:
            zf.extractall(tmpdir)
        converter = next(
            path for path in tmpdir.iterdir()
            if path.is_dir() and (path / "parse.rb").exists()
        )
        xml_dir = converter / "data.gov.ro"
        shutil.rmtree(xml_dir, ignore_errors=True)
        xml_dir.mkdir()
        for resource in romanian_resources(source):
            download(resource["url"], xml_dir / f"{resource['id']}.xml")
        subprocess.run([ruby, "parse.rb"], cwd=converter, check=True)
        gtfs_dir = converter / "gtfs-out"
        gtfs_files = sorted(gtfs_dir.glob("*.txt"))
        if not gtfs_files:
            raise RuntimeError("The Romanian converter produced no GTFS files")
        tempfile = outfile.with_suffix(".zip.part")
        try:
            with ZipFile(tempfile, "w", compression=ZIP_DEFLATED) as zf:
                for file in gtfs_files:
                    zf.write(file, file.name)
            tempfile.replace(outfile)
        finally:
            tempfile.unlink(missing_ok=True)
    return publish_feed(outfile, "Romania")


def convert_trenitalia(source):
    outfile = C.DATA / "gtfs/feeds/ext-Trenitalia.zip"
    if outfile.exists():
        validate_gtfs(outfile)
    else:
        from trenitalia import process_trenitalia
        process_trenitalia(source["url"])


#%% Run all external feed acquisitions
if __name__ == "__main__":
    errors = []
    ## Direct downloads
    C.log("Downloading external feeds using direct download links")
    try:
        for name, url in (pbar := tqdm(C.URLS["gtfs-feeds"].items())):
            pbar.set_description(name)
            outfile = FEEDS_DIR / f"ext-{name}.zip"
            if not outfile.exists():
                download(url, outfile)
                publish_feed(outfile, name)
    except Exception as e:
        C.error(f"Direct feeds: {e}")
        errors.append(f"Direct feeds: {e}")
    ## Manual downloads
    for name, handler in {
        "BDZ": download_bg_rail,
        "RO": download_romania,
        "Trenitalia": convert_trenitalia,
    }.items():
        try:
            handler(C.URLS["gtfs-edge-cases"][name])
        except Exception as e:
            C.error(f"{name}: {e}")
            errors.append(f"{name}: {e}")
    if errors:
        raise RuntimeError("\n".join(errors))
