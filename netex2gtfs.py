#!/usr/bin/env python3
"""
Convert Italian-profile NeTEx (PublicationDelivery) to a minimal GTFS feed.

Outputs GTFS ZIP containing:
- agency.txt
- stops.txt
- routes.txt
- trips.txt
- stop_times.txt
- calendar.txt
- feed_info.txt

CAVEAT:
- This script sets all service_ids active every day in the ValidBetween range.
  This is "OD-research-friendly" but not perfect for exact date-specific service.

Usage:
  python netex_to_gtfs_min.py netex_in/trenitalia.xml trenitalia_gtfs.zip
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import gzip
from lxml import etree
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve


NETEX_NS = "http://www.netex.org.uk/netex"
GML_NS = "http://www.opengis.net/gml/3.2"
NS = {"n": NETEX_NS, "gml": GML_NS} 


def localname(tag: str) -> str:
    return etree.QName(tag).localname


def text_or_none(el: Optional[etree._Element]) -> Optional[str]:
    if el is None:
        return None
    t = (el.text or "").strip()
    return t if t else None


def strip_time(ts: Optional[str]) -> Optional[str]:
    """
    Convert strings like:
      2026-01-23T14:08:34Z
      2026-01-23T14:08:34.000+02:00
      14:08:34
    into GTFS time HH:MM:SS.
    """
    if not ts:
        return None
    ts = ts.strip()
    # If it's ISO datetime, take time part
    m = re.search(r"T(\d{2}:\d{2}:\d{2})", ts)
    if m:
        return m.group(1)
    # If it already looks like HH:MM:SS
    m = re.match(r"^(\d{2}:\d{2}:\d{2})", ts)
    if m:
        return m.group(1)
    return None


def parse_iso_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    d = d.strip()
    # allow datetime too
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", d)
    if not m:
        return None
    y, mo, da = m.group(1).split("-")
    return date(int(y), int(mo), int(da))


def find_first(el: etree._Element, paths: Iterable[str]) -> Optional[etree._Element]:
    for p in paths:
        got = el.find(p, namespaces=NS)
        if got is not None:
            return got
    return None


def get_id(el: etree._Element) -> Optional[str]:
    return el.get("id") or el.get("{http://www.w3.org/XML/1998/namespace}id")


def get_ref(el: Optional[etree._Element]) -> Optional[str]:
    if el is None:
        return None
    return el.get("ref") or el.get("id") or el.text


def extract_lat_lon(el: etree._Element) -> Tuple[Optional[float], Optional[float]]:
    """
    Robust extraction of coordinates from Italian-profile NeTEx StopPlace/Quay/etc.
    Tries Latitude/Longitude elements and common GML Point/pos structures.
    Returns (lat, lon).
    """
    # 1) Direct Latitude/Longitude
    lat_el = el.xpath(".//*[local-name()='Latitude'][1]")
    lon_el = el.xpath(".//*[local-name()='Longitude'][1]")
    if lat_el and lon_el:
        try:
            return float((lat_el[0].text or "").strip()), float((lon_el[0].text or "").strip())
        except ValueError:
            pass

    # 2) GML pos (often within Centroid / Location / gml:Point)
    pos_el = el.xpath(".//*[local-name()='pos'][1]")
    if pos_el:
        pos = (pos_el[0].text or "").strip()
        parts = pos.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                a, b = float(parts[0]), float(parts[1])
                # Heuristic to decide order:
                # If one value is outside latitude bounds, it's longitude.
                if abs(a) > 90 and abs(b) <= 90:
                    lon, lat = a, b
                elif abs(b) > 90 and abs(a) <= 90:
                    lat, lon = a, b
                else:
                    # Many EU feeds use "lat lon"
                    lat, lon = a, b
                return lat, lon
            except ValueError:
                pass

    # 3) Old-style gml:coordinates "lon,lat" or "lat,lon"
    coords_el = el.xpath(".//*[local-name()='coordinates'][1]")
    if coords_el:
        txt = (coords_el[0].text or "").strip()
        # can be "lon,lat" or "lat,lon"
        parts = re.split(r"[ ,]+", txt)
        if len(parts) >= 2:
            try:
                a = float(parts[0].replace(",", "."))
                b = float(parts[1].replace(",", "."))
                if abs(a) > 90 and abs(b) <= 90:
                    lon, lat = a, b
                else:
                    lat, lon = a, b
                return lat, lon
            except ValueError:
                pass

    return None, None



@dataclass
class StopRow:
    stop_id: str
    stop_name: str
    stop_lat: Optional[float]
    stop_lon: Optional[float]
    location_type: int = 0
    parent_station: Optional[str] = None


@dataclass
class RouteRow:
    route_id: str
    agency_id: str
    route_short_name: str
    route_long_name: str
    route_type: int = 2  # rail


@dataclass
class TripRow:
    route_id: str
    service_id: str
    trip_id: str
    trip_headsign: Optional[str] = None


def pass1_extract_valid_between(xml_path: Path) -> Tuple[date, date]:
    """
    Extract ValidBetween FromDate/ToDate from the first CompositeFrame encountered.
    Fallback to a reasonable range if not found.
    """
    from_d = None
    to_d = None

    for event, elem in etree.iterparse(str(xml_path), events=("end",), huge_tree=True):
        if localname(elem.tag) == "CompositeFrame":
            from_el = elem.find(".//n:ValidBetween/n:FromDate", namespaces=NS)
            to_el = elem.find(".//n:ValidBetween/n:ToDate", namespaces=NS)
            from_d = parse_iso_date(text_or_none(from_el))
            to_d = parse_iso_date(text_or_none(to_el))
            elem.clear()
            break
        elem.clear()

    # Fallbacks
    if from_d is None:
        from_d = date(2026, 1, 1)
    if to_d is None:
        to_d = date(2026, 12, 31)
    return from_d, to_d


def pass2_extract_stops(xml_path: Path) -> Dict[str, StopRow]:
    """
    Build stops from StopPlace, Quay, ScheduledStopPoint.
    We'll emit all as location_type=0 (stops) except StopPlace can be parent station (location_type=1).
    Since counts are equal (1902), often IDs correspond; still, we keep whichever provides coords/names.
    """
    stops: Dict[str, StopRow] = {}
    stopplace_coords: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def upsert_stop(row: StopRow):
        existing = stops.get(row.stop_id)
        if existing is None:
            stops[row.stop_id] = row
            return
        # Merge: prefer non-empty name and non-null coords
        name = existing.stop_name if existing.stop_name else row.stop_name
        lat = existing.stop_lat if existing.stop_lat is not None else row.stop_lat
        lon = existing.stop_lon if existing.stop_lon is not None else row.stop_lon
        parent = existing.parent_station if existing.parent_station else row.parent_station
        loc_type = existing.location_type if existing.location_type != 0 else row.location_type
        stops[row.stop_id] = StopRow(
            stop_id=row.stop_id,
            stop_name=name,
            stop_lat=lat,
            stop_lon=lon,
            location_type=loc_type,
            parent_station=parent
        )

    for event, elem in etree.iterparse(str(xml_path), events=("end",), huge_tree=True):
        ln = localname(elem.tag)

        if ln == "StopPlace":
            sid = get_id(elem)
            if sid:
                name = text_or_none(find_first(elem, ["./n:Name", ".//n:Name"]))
                if not name:
                    name = sid
                lat, lon = extract_lat_lon(elem)
                stopplace_coords[sid] = (lat, lon)
                # StopPlace as parent station
                upsert_stop(StopRow(
                    stop_id=sid,
                    stop_name=name,
                    stop_lat=lat,
                    stop_lon=lon,
                    location_type=1,
                    parent_station=None
                ))

        elif ln == "Quay":
            qid = get_id(elem)
            if qid:
                name = text_or_none(find_first(elem, ["./n:Name", ".//n:Name"])) or qid
                lat, lon = extract_lat_lon(elem)
                # If quay has no coords, try to inherit via a StopPlaceRef if present
                parent_ref = get_ref(find_first(elem, ["./n:StopPlaceRef", ".//n:StopPlaceRef"]))
                if (lat is None or lon is None) and parent_ref and parent_ref in stopplace_coords:
                    lat, lon = stopplace_coords[parent_ref]
                upsert_stop(StopRow(
                    stop_id=qid,
                    stop_name=name,
                    stop_lat=lat,
                    stop_lon=lon,
                    location_type=0,
                    parent_station=parent_ref
                ))

        elif ln == "ScheduledStopPoint":
            spid = get_id(elem)
            if spid:
                name = text_or_none(find_first(elem, ["./n:Name", ".//n:Name"])) or spid
                lat, lon = extract_lat_lon(elem)
                # Try parent station ref
                parent_ref = get_ref(find_first(elem, ["./n:StopPlaceRef", ".//n:StopPlaceRef"]))
                if (lat is None or lon is None) and parent_ref and parent_ref in stopplace_coords:
                    lat, lon = stopplace_coords[parent_ref]
                upsert_stop(StopRow(
                    stop_id=spid,
                    stop_name=name,
                    stop_lat=lat,
                    stop_lon=lon,
                    location_type=0,
                    parent_station=parent_ref
                ))

        elem.clear()

    return stops


def pass3_extract_routes(xml_path: Path, agency_id: str) -> Dict[str, RouteRow]:
    routes: Dict[str, RouteRow] = {}

    # We prefer Line objects as routes in GTFS
    for event, elem in etree.iterparse(str(xml_path), events=("end",), huge_tree=True):
        ln = localname(elem.tag)
        if ln == "Line":
            lid = get_id(elem)
            if lid:
                name = text_or_none(find_first(elem, ["./n:Name", ".//n:Name"])) or lid
                public_code = text_or_none(find_first(elem, ["./n:PublicCode", ".//n:PublicCode"]))
                short = public_code or name
                routes[lid] = RouteRow(
                    route_id=lid,
                    agency_id=agency_id,
                    route_short_name=short[:50],
                    route_long_name=name[:1000]
                )
        elem.clear()

    # Fallback: if no Lines, use Route elements
    if not routes:
        for event, elem in etree.iterparse(str(xml_path), events=("end",), huge_tree=True):
            if localname(elem.tag) == "Route":
                rid = get_id(elem)
                if rid:
                    name = text_or_none(find_first(elem, ["./n:Name", ".//n:Name"])) or rid
                    routes[rid] = RouteRow(
                        route_id=rid,
                        agency_id=agency_id,
                        route_short_name=name[:50],
                        route_long_name=name[:1000]
                    )
            elem.clear()

    return routes


def pass4_write_trips_and_stop_times(
    xml_path: Path,
    routes: Dict[str, RouteRow],
    trips_csv: Path,
    stop_times_csv: Path,
) -> Tuple[int, int]:
    num_trips = 0
    num_stop_times = 0

    def route_fallback() -> str:
        return next(iter(routes.keys())) if routes else "UNKNOWN_ROUTE"

    in_journey = False
    cur_trip_id: Optional[str] = None
    cur_route_id: Optional[str] = None
    cur_service_id: Optional[str] = None
    cur_headsign: Optional[str] = None
    cur_seq = 0

    # Passing-time context
    in_pt = False
    pt_stop_ref: Optional[str] = None
    pt_arr: Optional[str] = None
    pt_dep: Optional[str] = None

    with trips_csv.open("w", newline="", encoding="utf-8") as ft, \
         stop_times_csv.open("w", newline="", encoding="utf-8") as fs:
        trips_w = csv.writer(ft)
        st_w = csv.writer(fs)
        trips_w.writerow(["route_id", "service_id", "trip_id", "trip_headsign"])
        st_w.writerow(["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])

        context = etree.iterparse(str(xml_path), events=("start", "end"), huge_tree=True)

        for event, elem in context:
            tag = localname(elem.tag)

            # --- Journey context (note: file uses ServiceJourney elements with ids that start IT::VehicleJourney...)
            if event == "start" and tag in {"ServiceJourney", "VehicleJourney"}:
                in_journey = True
                cur_trip_id = get_id(elem)
                cur_route_id = None
                cur_service_id = None
                cur_headsign = None
                cur_seq = 0
                # don't clear on start
                continue

            # --- Passing time context
            if in_journey and event == "start" and tag == "TimetabledPassingTime":
                in_pt = True
                pt_stop_ref = None
                pt_arr = None
                pt_dep = None
                continue

            if in_journey and in_pt and event == "end":
                # Stop refs commonly appear as elements with @ref
                if tag in {
                    "StopPointInJourneyPatternRef",
                    "ScheduledStopPointRef",
                    "StopPointRef",
                    "QuayRef",
                    "StopPlaceRef",
                }:
                    r = elem.get("ref")
                    if r and pt_stop_ref is None:
                        pt_stop_ref = r

                # Times: can be ArrivalTime/DepartureTime or nested structures.
                elif tag == "ArrivalTime":
                    pt_arr = strip_time((elem.text or "").strip())
                elif tag == "DepartureTime":
                    pt_dep = strip_time((elem.text or "").strip())

                # Some feeds use <Arrival><Time>HH:MM:SS</Time></Arrival>
                elif tag == "Time":
                    # only use this if we haven't already captured arr/dep
                    t = strip_time((elem.text or "").strip())
                    if t:
                        # Heuristic: first Time encountered inside PT becomes arrival, second becomes departure
                        if pt_arr is None:
                            pt_arr = t
                        elif pt_dep is None:
                            pt_dep = t

                elif tag == "TimetabledPassingTime":
                    # End of passing time: write stop_time row if we have enough
                    if cur_trip_id and pt_stop_ref and (pt_arr or pt_dep):
                        arr = pt_arr or pt_dep
                        dep = pt_dep or pt_arr
                        cur_seq += 1
                        st_w.writerow([cur_trip_id, arr, dep, pt_stop_ref, cur_seq])
                        num_stop_times += 1

                    # reset PT context
                    in_pt = False
                    pt_stop_ref = None
                    pt_arr = None
                    pt_dep = None

                    # safe to clear this element now
                    elem.clear()
                    continue

                # While inside PT, do NOT blindly clear children before end-of-PT.
                # But it's still okay to clear leaf nodes AFTER we've read them:
                elem.clear()
                continue

            # --- Still inside journey: capture metadata refs
            if in_journey and event == "end" and not in_pt:
                if tag == "LineRef":
                    r = elem.get("ref")
                    if r and r in routes:
                        cur_route_id = r
                elif tag == "RouteRef":
                    r = elem.get("ref")
                    if r and (cur_route_id is None) and (r in routes):
                        cur_route_id = r
                elif tag == "DayTypeRef":
                    r = elem.get("ref")
                    if r:
                        cur_service_id = r
                elif tag in {"DestinationDisplay", "Name"} and cur_headsign is None:
                    t = (elem.text or "").strip()
                    if t:
                        cur_headsign = t

                elif tag in {"ServiceJourney", "VehicleJourney"}:
                    # journey end: write trip
                    if cur_trip_id:
                        rid = cur_route_id if cur_route_id else route_fallback()
                        sid = cur_service_id or "ALL_DAYS"
                        trips_w.writerow([rid, sid, cur_trip_id, cur_headsign or ""])
                        num_trips += 1

                    in_journey = False
                    cur_trip_id = None
                    cur_route_id = None
                    cur_service_id = None
                    cur_headsign = None
                    cur_seq = 0

                    elem.clear()
                    continue

                elem.clear()
                continue

            # --- Outside journey: can safely clear
            if event == "end":
                elem.clear()

    return num_trips, num_stop_times



def write_gtfs_zip(
    out_zip: Path,
    agency_id: str,
    agency_name: str,
    agency_url: str,
    agency_timezone: str,
    from_d: date,
    to_d: date,
    stops: Dict[str, StopRow],
    routes: Dict[str, RouteRow],
    trips_csv: Path,
    stop_times_csv: Path,
) -> None:
    tmpdir = trips_csv.parent

    agency_txt = tmpdir / "agency.txt"
    stops_txt = tmpdir / "stops.txt"
    routes_txt = tmpdir / "routes.txt"
    calendar_txt = tmpdir / "calendar.txt"
    feed_info_txt = tmpdir / "feed_info.txt"

    # agency.txt
    with agency_txt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agency_id", "agency_name", "agency_url", "agency_timezone"])
        w.writerow([agency_id, agency_name, agency_url, agency_timezone])

    # stops.txt
    with stops_txt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"])
        for s in stops.values():
            w.writerow([
                s.stop_id,
                s.stop_name,
                "" if s.stop_lat is None else f"{s.stop_lat:.8f}",
                "" if s.stop_lon is None else f"{s.stop_lon:.8f}",
                s.location_type,
                s.parent_station or ""
            ])

    # routes.txt
    with routes_txt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
        for r in routes.values():
            w.writerow([r.route_id, r.agency_id, r.route_short_name, r.route_long_name, r.route_type])

    # calendar.txt
    # We declare every service_id active all days across ValidBetween.
    # To get the list of service_ids, read trips_csv (streaming is fine; it's small-ish).
    service_ids = set()
    with trips_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            service_ids.add(row["service_id"])

    with calendar_txt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "start_date", "end_date"])
        sd = from_d.strftime("%Y%m%d")
        ed = to_d.strftime("%Y%m%d")
        for sid in sorted(service_ids):
            w.writerow([sid, 1, 1, 1, 1, 1, 1, 1, sd, ed])

    # feed_info.txt
    with feed_info_txt.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_start_date", "feed_end_date"])
        w.writerow([agency_name, agency_url, "it", from_d.strftime("%Y%m%d"), to_d.strftime("%Y%m%d")])

    # Build zip
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(agency_txt, arcname="agency.txt")
        z.write(stops_txt, arcname="stops.txt")
        z.write(routes_txt, arcname="routes.txt")
        z.write(trips_csv, arcname="trips.txt")
        z.write(stop_times_csv, arcname="stop_times.txt")
        z.write(calendar_txt, arcname="calendar.txt")
        z.write(feed_info_txt, arcname="feed_info.txt")


def convert_netex_to_gtfs(xml_path: Path, out_zip: Path):
    # if len(sys.argv) != 3:
    #     print("Usage: python netex_to_gtfs_min.py <netex.xml> <out_gtfs.zip>", file=sys.stderr)
    #     sys.exit(2)
    # xml_path = Path(sys.argv[1]).expanduser().resolve()
    # out_zip = Path(sys.argv[2]).expanduser().resolve()

    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    tmpdir = out_zip.parent / (out_zip.stem + "_tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading validity window...")
    from_d, to_d = pass1_extract_valid_between(xml_path)
    print(f"  ValidBetween: {from_d} → {to_d}")

    print(f"[2/4] Extracting stops...")
    stops = pass2_extract_stops(xml_path)
    print(f"  stops extracted: {len(stops):,}")

    print(f"[3/4] Extracting routes...")
    agency_id = "TRENITALIA"
    routes = pass3_extract_routes(xml_path, agency_id=agency_id)
    print(f"  routes extracted: {len(routes):,}")

    trips_csv = tmpdir / "trips.txt"
    stop_times_csv = tmpdir / "stop_times.txt"

    print(f"[4/4] Writing trips & stop_times (streaming)...")
    n_trips, n_st = pass4_write_trips_and_stop_times(
        xml_path=xml_path, routes=routes, trips_csv=trips_csv,
        stop_times_csv=stop_times_csv)
    print(f"  trips: {n_trips:,}")
    print(f"  stop_times: {n_st:,}")

    print(f"[+] Building GTFS zip: {out_zip.name}")
    write_gtfs_zip(
        out_zip=out_zip, agency_id=agency_id,
        agency_name="Trenitalia",
        agency_url="https://www.trenitalia.com",
        agency_timezone="Europe/Rome",
        from_d=from_d, to_d=to_d, stops=stops, routes=routes,
        trips_csv=trips_csv, stop_times_csv=stop_times_csv
    )
    shutil.rmtree(tmpdir)
    print("[OK] Done.")
    print(f"GTFS written to: {out_zip}")


def get_station_coords(stops_file):
    ## NeTEx train stops
    stops = pd.read_csv(stops_file, usecols=["stop_id", "location_type", "parent_station"])
    stops["code"] = stops["stop_id"].str.split(":").str[-1].str[2:].astype(int)
    ## Train stations from Trainline database
    url = "https://raw.githubusercontent.com/trainline-eu/stations/refs/heads/master/stations.csv"
    stns = (pd.read_csv(url, sep=";", usecols=["name", "uic", "country", "longitude", "latitude"])
        .query("country == 'IT'").dropna()
        .rename(columns={"longitude": "lon", "latitude": "lat"})
        .astype({"uic": int}).astype({"uic": str})
        .reset_index(drop=True)[["uic", "name", "lon", "lat"]])
    stns["code"] = stns["uic"].str[2:].astype(int)
    ## Get coordinates from Trainline table to NeTEx table
    stops = (stops.merge(stns, on="code")
             .rename(columns=dict(lon="stop_lon", lat="stop_lat", name="stop_name"))
             ["stop_id stop_name stop_lat stop_lon location_type parent_station".split()])
    return stops


def build_spjp_map(netex_xml_path: str) -> dict[str, str]:
    """
    Returns dict: StopPointInJourneyPattern.id -> referenced physical stop id
    (prefers QuayRef, then ScheduledStopPointRef, then StopPlaceRef).
    """
    spjp_to_stop = {}
    in_spjp = False
    cur_spjp_id = None
    cur_ref = None
    ctx = etree.iterparse(netex_xml_path, events=("start", "end"), huge_tree=True)
    for event, el in ctx:
        tag = localname(el.tag)
        if event == "start" and tag == "StopPointInJourneyPattern":
            in_spjp = True
            cur_spjp_id = el.get("id")
            cur_ref = None
            continue
        if in_spjp and event == "end":
            # Prefer platform-level IDs when available
            if tag in {"QuayRef", "ScheduledStopPointRef", "StopPlaceRef"}:
                r = el.get("ref")
                if r:
                    # priority: Quay > ScheduledStopPoint > StopPlace
                    if cur_ref is None:
                        cur_ref = r
                    else:
                        # upgrade if we currently have a lower-priority ref
                        prio = {"StopPlaceRef": 0, "ScheduledStopPointRef": 1, "QuayRef": 2}
                        if prio[tag] > prio.get(cur_ref.split(":")[2] if "::" in cur_ref else "", -1):
                            cur_ref = r
            elif tag == "StopPointInJourneyPattern":
                if cur_spjp_id and cur_ref:
                    spjp_to_stop[cur_spjp_id] = cur_ref
                in_spjp = False
                cur_spjp_id = None
                cur_ref = None
                el.clear()
                continue
            el.clear()
        if event == "end" and not in_spjp:
            el.clear()
    return spjp_to_stop


if __name__ == "__main__":
    ROOT = Path("/Users/rajatverma/Documents/research-data/3mars/gtfs")
    ## Download the NeTEx feed from the CCISS website
    netex_url = "https://www.cciss.it/nap/mmtis/public/api/v1/download/blob/Asset/1080596/checkedResource"
    if not (xml_gz_path := ROOT / "trenitalia.xml.gz").exists():
        urlretrieve(netex_url, xml_gz_path)
    ## Unzip the NeTEx XML file
    if not (xml_path := ROOT / "trenitalia-netex.xml").exists():
        with gzip.open(xml_gz_path, 'rb') as file_in:
            with open(xml_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
    ## Convert NeTEx to GTFS and unzip zip
    if not (zip_path := ROOT / "trenitalia-gtfs.zip").exists():
        convert_netex_to_gtfs(xml_path, zip_path)
    feed_name = "man-Trenitalia_netex"
    if not (gtfs_dir := ROOT / "raw" / feed_name).exists():
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(gtfs_dir)
    ## Clean the stops and stop_times tables
    if gtfs_dir.exists():
        ## Obtain station names from the Trainline dataset
        stops = get_station_coords(stops_file := gtfs_dir / "stops.txt")
        stops.to_csv(stops_file, index=False)
        ## Map stoptimes.stop_id to stops.stop_id
        stops = pd.read_csv(stops_file := gtfs_dir / "stops.txt")
        times = pd.read_csv(times_file := gtfs_dir / "stop_times.txt")
        spjp2stop = build_spjp_map(xml_path)
        print(len(spjp2stop), list(spjp2stop.items())[0])
        times["stop_id"] = times["stop_id"].map(spjp2stop)
        times.dropna(subset="stop_id", inplace=True)
        stops = stops.merge(times["stop_id"].drop_duplicates(), on="stop_id")
        stops.to_csv(stops_file, index=False)
        times.to_csv(times_file, index=False)
        pass
