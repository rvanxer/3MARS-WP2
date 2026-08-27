"""Convert the six prepared intercity tables to a GTFS Schedule feed.

Timetable values have already been normalised to fixed GMT in intercity.py.
Journeys that consequently begin before 00:00 are moved to the preceding GMT
service day and represented with GTFS times above 24:00.

Shapes are assembled from modal geometries between consecutive stations. A
shape is emitted only when geometry is available for every segment of a line;
trips on incomplete lines remain valid without a shape_id.

The source tables do not contain operator websites, so MobilityDatabase's
catalogue URL is used as the required agency_url. calendar.txt is omitted and
all active service days are written to calendar_dates.txt.
"""

#%% Imports
from contextlib import contextmanager
from datetime import date
import io
from pathlib import Path
import re
import zipfile

import geopandas as gpd
from numba import njit
import numpy as np
import pandas as pd
from pyproj import Geod

import config as C


I8, I16, I32, I64 = np.int8, np.int16, np.int32, np.int64
UI8, UI16 = np.uint8, np.uint16
F64 = np.float64

OUTPUT = C.DATA / "gtfs/eu-intercity.gtfs.zip"
AGENCY_URL = "https://database.mobilitydata.org/"
AGENCY_TIMEZONE = "GMT"
CSV_CHUNK_SIZE = 500_000
CALENDAR_CHUNK_SIZE = 2_048
SHAPE_CHUNK_SIZE = 250_000
GEOD = Geod(ellps="WGS84")

REQUIRED_COLUMNS = {
    "stations": {"stn", "name", "geometry"},
    "lines": {"line", "agency", "rail", "stn"},
    "journeys": {"jrn", "line", "dateset"},
    "datesets": {"dateset"},
    "timetable": {"jrn", "stn", "arr", "dep"},
    "segments": {"src", "trg", "mode", "line", "geometry"},
}


#%% Validation helpers
def require_columns(df: pd.DataFrame, table: str) -> None:
    missing = REQUIRED_COLUMNS[table].difference(df.columns)
    if missing:
        raise ValueError(f"{table} is missing columns: {sorted(missing)}")


def validate_source_tables(
    stns: gpd.GeoDataFrame,
    lines: pd.DataFrame,
    journeys: pd.DataFrame,
    datesets: pd.DataFrame,
    tt: pd.DataFrame,
    segments: gpd.GeoDataFrame,
) -> None:
    for name, table in {
        "stations": stns,
        "lines": lines,
        "journeys": journeys,
        "datesets": datesets,
        "timetable": tt,
        "segments": segments,
    }.items():
        require_columns(table, name)

    for table, key in [
        (stns, "stn"), (lines, "line"),
        (journeys, "jrn"), (datesets, "dateset"),
    ]:
        if table[key].duplicated().any():
            raise ValueError(f"{key} is not unique")

    if lines[["line", "agency", "rail", "stn"]].isna().any().any():
        raise ValueError("lines contains missing required values")
    if not lines["stn"].str.len().ge(2).all():
        raise ValueError("every line must contain at least two stations")
    if not journeys["line"].isin(lines["line"]).all():
        raise ValueError("journeys references an unknown line")
    if not journeys["dateset"].isin(datesets["dateset"]).all():
        raise ValueError("journeys references an unknown dateset")

    if stns.crs is None:
        raise ValueError("stations has no coordinate reference system")
    if stns.geometry.isna().any() or stns.geometry.is_empty.any():
        raise ValueError("stations contains missing or empty geometry")
    if not stns.geometry.geom_type.eq("Point").all():
        raise ValueError("every station geometry must be a Point")

    if tt[["jrn", "stn", "arr", "dep"]].isna().any().any():
        raise ValueError("timetable contains missing required values")
    if (tt["arr"] > tt["dep"]).any():
        raise ValueError("timetable contains arrivals later than departures")
    if not tt["jrn"].isin(journeys["jrn"]).all():
        raise ValueError("timetable references an unknown journey")
    if not tt["stn"].isin(stns["stn"]).all():
        raise ValueError("timetable references an unknown station")
    if (tt["jrn"].to_numpy()[1:] < tt["jrn"].to_numpy()[:-1]).any():
        raise ValueError("timetable journeys are not stored in order")

    if segments.crs is None or segments.crs.to_epsg() != 4326:
        raise ValueError("segments must use WGS84 coordinates")
    if segments.duplicated(["src", "trg", "mode"]).any():
        raise ValueError("segment mode and station pair is not unique")
    if not segments["mode"].isin(["Bus", "Rail"]).all():
        raise ValueError("segments contains an unknown mode")
    invalid = segments.geometry.notna() & ~segments.geometry.is_valid
    if invalid.any():
        raise ValueError("segments contains invalid geometry")


#%% Core GTFS tables
def prepare_agencies_lines(
    stns: gpd.GeoDataFrame, lines: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agency_name = lines["agency"].astype("string").str.strip()
    agency_name = agency_name.fillna("Unknown agency").replace("", "Unknown agency")
    agency_code, agency_names = pd.factorize(agency_name, sort=True)

    lines = lines.copy()
    lines["agency_id"] = agency_code.astype(I32) + 1
    agencies = pd.DataFrame({
        "agency_id": np.arange(1, len(agency_names) + 1, dtype=I32),
        "agency_name": agency_names,
        "agency_url": AGENCY_URL,
        "agency_timezone": AGENCY_TIMEZONE,
    })
    stn_name = (stns.set_index("stn")["name"].astype(str)
                .str.split(" | ").str[0].str.strip())
    origins = lines["stn"].str[0].map(stn_name)
    destinations = lines["stn"].str[-1].map(stn_name)
    if origins.isna().any() or destinations.isna().any():
        raise ValueError("a line endpoint is absent from stations")
    routes = pd.DataFrame({
        "route_id": lines["line"].to_numpy(),
        "agency_id": lines["agency_id"].to_numpy(),
        "route_short_name": "IC-" + lines["line"].astype(str).str.zfill(5),
        "route_long_name": origins + " – " + destinations,
        "route_type": np.where(lines["rail"], 2, 3).astype(UI8),
    })
    return agencies, routes, lines


def prepare_stops(stns: gpd.GeoDataFrame) -> pd.DataFrame:
    stns = stns.to_crs(4326).set_index("stn").sort_index()
    stops = pd.DataFrame({
        "stop_id": stns.index,
        "stop_name": stns["name"].astype(str).str.strip().to_numpy(),
        "stop_lat": stns.geometry.y.to_numpy(),
        "stop_lon": stns.geometry.x.to_numpy(),
        "location_type": UI8(0),
    })
    if not stops["stop_lat"].between(-90, 90).all():
        raise ValueError("station latitude lies outside WGS84 bounds")
    if not stops["stop_lon"].between(-180, 180).all():
        raise ValueError("station longitude lies outside WGS84 bounds")
    return stops


def prepare_trips_services(
    journeys: pd.DataFrame,
    tt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    trip_values = tt["jrn"].to_numpy(dtype=I32, copy=False)
    starts = np.flatnonzero(
        np.r_[True, trip_values[1:] != trip_values[:-1]]
    ).astype(I64)
    counts = np.diff(np.r_[starts, len(tt)]).astype(I32)
    trip_ids = trip_values[starts]

    trip_meta = journeys.set_index("jrn").reindex(trip_ids)
    if trip_meta[["line", "dateset"]].isna().any().any():
        raise ValueError("timetable and journeys do not contain the same journey IDs")
    if len(trip_meta) != len(journeys):
        raise ValueError("some journeys have no timetable events")

    first_arrival = tt["arr"].to_numpy(dtype=I32, copy=False)[starts]
    day_shift = np.where(first_arrival < 0, -1, 0).astype(I8)
    if (first_arrival < -1440).any():
        raise ValueError("a journey begins more than one day before its service date")
    trip_meta["day_shift"] = day_shift

    services = (
        trip_meta[["dateset", "day_shift"]]
        .drop_duplicates()
        .sort_values(["dateset", "day_shift"], ignore_index=True)
    )
    services["service_id"] = np.arange(1, len(services) + 1, dtype=I32)
    trip_meta = (
        trip_meta.reset_index()
        .merge(services, on=["dateset", "day_shift"], validate="many_to_one")
        .set_index("jrn")
        .reindex(trip_ids)
    )
    trips = pd.DataFrame({
        "route_id": trip_meta["line"].to_numpy(dtype=I32),
        "service_id": trip_meta["service_id"].to_numpy(dtype=I32),
        "trip_id": trip_ids,
    })
    return trips, services, starts, counts, trip_meta["line"].to_numpy(dtype=I32)


#%% Shape assembly
def append_coordinate(output: list[tuple[float, float]], coordinate) -> None:
    point = (round(float(coordinate[0]), 7), round(float(coordinate[1]), 7))
    if not output or point != output[-1]:
        output.append(point)


def orient_geometry_parts(geometry, start) -> list[np.ndarray]:
    parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
    coordinates = [np.asarray(part.coords, dtype=F64)[:, :2] for part in parts]
    current = np.asarray(start, dtype=F64)
    output = []
    while coordinates:
        best_part, reverse, best_distance = 0, False, np.inf
        for i, part in enumerate(coordinates):
            first = np.sum((part[0] - current) ** 2)
            last = np.sum((part[-1] - current) ** 2)
            if first < best_distance:
                best_part, reverse, best_distance = i, False, first
            if last < best_distance:
                best_part, reverse, best_distance = i, True, last
        part = coordinates.pop(best_part)
        if reverse:
            part = part[::-1]
        output.append(part)
        current = part[-1]
    return output


def assemble_shape(
    station_ids: tuple[int, ...],
    mode: str,
    station_xy: dict[int, tuple[float, float]],
    segment_lookup: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    coordinates = []
    append_coordinate(coordinates, station_xy[station_ids[0]])
    station_positions = [0]
    for src, trg in zip(station_ids[:-1], station_ids[1:]):
        segment = segment_lookup.get((src, trg, mode))
        if segment is None or segment.is_empty:
            return None
        for part in orient_geometry_parts(segment, coordinates[-1]):
            for coordinate in part:
                append_coordinate(coordinates, coordinate)
        append_coordinate(coordinates, station_xy[trg])
        station_positions.append(len(coordinates) - 1)

    coordinates = np.asarray(coordinates, dtype=F64)
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        return None
    _, _, distances = GEOD.inv(
        coordinates[:-1, 0], coordinates[:-1, 1],
        coordinates[1:, 0], coordinates[1:, 1],
    )
    if not np.isfinite(distances).all() or (distances <= 0).any():
        return None
    cumulative = np.r_[0.0, np.cumsum(distances / 1000)]
    return coordinates, cumulative[np.asarray(station_positions)]


def write_shapes(
    feed: zipfile.ZipFile,
    lines: pd.DataFrame,
    segments: gpd.GeoDataFrame,
    stns: gpd.GeoDataFrame,
) -> tuple[np.ndarray, list[np.ndarray | None], dict[str, int]]:
    station_xy = {
        int(stn): (geometry.x, geometry.y)
        for stn, geometry in stns.to_crs(4326).set_index("stn").geometry.items()
    }
    segment_lookup = {
        (int(row.src), int(row.trg), row.mode): row.geometry
        for row in segments.itertuples()
        if row.geometry is not None
    }

    line_shape_ids = np.full(len(lines), -1, dtype=I32)
    line_stop_distances: list[np.ndarray | None] = [None] * len(lines)
    shape_cache: dict[tuple[bool, tuple[int, ...]], tuple[int, np.ndarray] | None] = {}
    next_shape_id = 1
    shape_rows = 0
    incomplete_lines = 0

    with zip_text_member(feed, "shapes.txt") as member:
        member.write(
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,"
            "shape_dist_traveled\n"
        )
        buffer = []
        buffer_rows = 0
        for position, row in enumerate(lines.itertuples(index=False)):
            station_ids = tuple(map(int, row.stn))
            key = (bool(row.rail), station_ids)
            cached = shape_cache.get(key, False)
            if cached is False:
                mode = "Rail" if row.rail else "Bus"
                assembled = assemble_shape(
                    station_ids, mode, station_xy, segment_lookup
                )
                if assembled is None:
                    shape_cache[key] = None
                    cached = None
                else:
                    coordinates, stop_distances = assembled
                    shape_id = next_shape_id
                    next_shape_id += 1
                    _, _, distances = GEOD.inv(
                        coordinates[:-1, 0], coordinates[:-1, 1],
                        coordinates[1:, 0], coordinates[1:, 1],
                    )
                    cumulative = np.r_[0.0, np.cumsum(distances / 1000)]
                    buffer.append(pd.DataFrame({
                        "shape_id": shape_id,
                        "shape_pt_lat": coordinates[:, 1],
                        "shape_pt_lon": coordinates[:, 0],
                        "shape_pt_sequence": np.arange(1, len(coordinates) + 1),
                        "shape_dist_traveled": cumulative,
                    }))
                    buffer_rows += len(coordinates)
                    shape_rows += len(coordinates)
                    cached = (shape_id, stop_distances)
                    shape_cache[key] = cached
            if cached is None:
                incomplete_lines += 1
            else:
                line_shape_ids[position] = cached[0]
                line_stop_distances[position] = cached[1]

            if buffer_rows >= SHAPE_CHUNK_SIZE:
                pd.concat(buffer, ignore_index=True).to_csv(
                    member, index=False, header=False, lineterminator="\n",
                    float_format="%.7f",
                )
                buffer, buffer_rows = [], 0
        if buffer:
            pd.concat(buffer, ignore_index=True).to_csv(
                member, index=False, header=False, lineterminator="\n",
                float_format="%.7f",
            )

    stats = {
        "shapes": next_shape_id - 1,
        "shape_points": shape_rows,
        "lines_without_complete_shape": incomplete_lines,
    }
    return line_shape_ids, line_stop_distances, stats


#%% Timetable expansion
@njit
def prepare_stop_time_arrays(
    stations: np.ndarray,
    arrivals: np.ndarray,
    departures: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    journey_line_rows: np.ndarray,
    line_starts: np.ndarray,
    line_counts: np.ndarray,
    line_stations: np.ndarray,
    line_distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    sequence = np.empty(len(stations), dtype=np.uint16)
    output_arrivals = np.empty(len(stations), dtype=np.int32)
    output_departures = np.empty(len(stations), dtype=np.int32)
    shape_distances = np.empty(len(stations), dtype=np.float64)
    valid = True
    for group in range(len(starts)):
        start = starts[group]
        count = counts[group]
        line_row = journey_line_rows[group]
        line_start = line_starts[line_row]
        if count != line_counts[line_row]:
            valid = False
            continue
        adjustment = 1440 if arrivals[start] < 0 else 0
        for position in range(count):
            row = start + position
            line_position = line_start + position
            if stations[row] != line_stations[line_position]:
                valid = False
            sequence[row] = position + 1
            output_arrivals[row] = arrivals[row] + adjustment
            output_departures[row] = departures[row] + adjustment
            shape_distances[row] = line_distances[line_position]
    return (
        sequence, output_arrivals, output_departures,
        shape_distances, valid,
    )


def build_stop_time_arrays(
    tt: pd.DataFrame,
    lines: pd.DataFrame,
    starts: np.ndarray,
    counts: np.ndarray,
    journey_lines: np.ndarray,
    line_stop_distances: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    line_ids = lines["line"].to_numpy(dtype=I32)
    line_row = pd.Series(np.arange(len(lines), dtype=I32), index=line_ids)
    journey_line_rows = line_row.reindex(journey_lines).to_numpy(dtype=I32)
    line_counts = lines["stn"].str.len().to_numpy(dtype=I32)
    line_starts = np.r_[0, np.cumsum(line_counts[:-1], dtype=I64)]
    line_stations = np.concatenate(lines["stn"].to_numpy()).astype(I32)
    line_distances = np.concatenate([
        distances if distances is not None else np.full(count, np.nan)
        for distances, count in zip(line_stop_distances, line_counts)
    ]).astype(F64)

    sequence, arrivals, departures, shape_distances, valid = (
        prepare_stop_time_arrays(
            tt["stn"].to_numpy(dtype=I32, copy=False),
            tt["arr"].to_numpy(dtype=I32, copy=False),
            tt["dep"].to_numpy(dtype=I32, copy=False),
            starts, counts, journey_line_rows,
            line_starts, line_counts, line_stations, line_distances,
        )
    )
    if not valid:
        raise ValueError("timetable station sequences do not match their lines")
    if (arrivals < 0).any() or (departures < 0).any():
        raise ValueError("adjusted GTFS times remain negative")
    same_trip = tt["jrn"].eq(tt["jrn"].shift())
    if (same_trip & pd.Series(arrivals).lt(pd.Series(departures).shift())).any():
        raise ValueError("stop events overlap within a journey")
    return sequence, arrivals, departures, shape_distances


#%% ZIP writers
@contextmanager
def zip_text_member(feed: zipfile.ZipFile, name: str):
    raw = feed.open(name, mode="w", force_zip64=True)
    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.flush()
        text.detach()
        raw.close()


def write_frame(feed: zipfile.ZipFile, name: str, frame: pd.DataFrame) -> None:
    with zip_text_member(feed, name) as member:
        frame.to_csv(member, index=False, lineterminator="\n")


def minutes_to_gtfs_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=I64)
    hours = np.char.zfill((values // 60).astype(str), 2)
    minutes = np.char.zfill((values % 60).astype(str), 2)
    return np.char.add(np.char.add(np.char.add(hours, ":"), minutes), ":00")


def write_stop_times(
    feed: zipfile.ZipFile,
    tt: pd.DataFrame,
    sequence: np.ndarray,
    arrivals: np.ndarray,
    departures: np.ndarray,
    shape_distances: np.ndarray,
) -> None:
    with zip_text_member(feed, "stop_times.txt") as member:
        member.write(
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence,"
            "shape_dist_traveled\n"
        )
        for start in range(0, len(tt), CSV_CHUNK_SIZE):
            end = min(start + CSV_CHUNK_SIZE, len(tt))
            output = pd.DataFrame({
                "trip_id": tt["jrn"].to_numpy()[start:end],
                "arrival_time": minutes_to_gtfs_time(arrivals[start:end]),
                "departure_time": minutes_to_gtfs_time(departures[start:end]),
                "stop_id": tt["stn"].to_numpy()[start:end],
                "stop_sequence": sequence[start:end],
                "shape_dist_traveled": shape_distances[start:end],
            })
            output.to_csv(
                member, index=False, header=False, lineterminator="\n",
                na_rep="", float_format="%.7f",
            )


def write_calendar_dates(
    feed: zipfile.ZipFile,
    datesets: pd.DataFrame,
    services: pd.DataFrame,
) -> tuple[int, str, str]:
    datesets = datesets.set_index("dateset")
    date_columns = np.asarray([str(column) for column in datesets.columns])
    if not all(re.fullmatch(r"\d{8}", value) for value in date_columns):
        raise ValueError("dateset columns must be YYYYMMDD dates")
    dates = pd.to_datetime(date_columns, format="%Y%m%d", errors="raise")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dateset date columns must be unique and chronological")
    date_values = dates.to_numpy(dtype="datetime64[D]")

    row_count = 0
    first_date, last_date = None, None
    with zip_text_member(feed, "calendar_dates.txt") as member:
        member.write("service_id,date,exception_type\n")
        for start in range(0, len(services), CALENDAR_CHUNK_SIZE):
            part = services.iloc[start:start + CALENDAR_CHUNK_SIZE]
            activity = datesets.loc[part["dateset"]].to_numpy(dtype=bool, copy=False)
            if not activity.any(axis=1).all():
                raise ValueError("a service used by trips has no active dates")
            active_rows, active_columns = np.nonzero(activity)
            shifts = part["day_shift"].to_numpy(dtype=I64)[active_rows]
            shifted_dates = (
                date_values[active_columns] + shifts.astype("timedelta64[D]")
            )
            date_strings = np.char.replace(
                np.datetime_as_string(shifted_dates, unit="D"), "-", ""
            )
            output = pd.DataFrame({
                "service_id": part["service_id"].to_numpy()[active_rows],
                "date": date_strings,
                "exception_type": UI8(1),
            })
            output.to_csv(member, index=False, header=False, lineterminator="\n")
            row_count += len(output)
            chunk_min, chunk_max = min(date_strings), max(date_strings)
            first_date = chunk_min if first_date is None else min(first_date, chunk_min)
            last_date = chunk_max if last_date is None else max(last_date, chunk_max)
    return row_count, first_date, last_date


def validate_relations(
    agencies: pd.DataFrame,
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    tt: pd.DataFrame,
    shape_ids: np.ndarray,
) -> None:
    if not routes["agency_id"].isin(agencies["agency_id"]).all():
        raise ValueError("routes contains an unknown agency_id")
    if not trips["route_id"].isin(routes["route_id"]).all():
        raise ValueError("trips contains an unknown route_id")
    if not tt["jrn"].isin(trips["trip_id"]).all():
        raise ValueError("stop_times contains an unknown trip_id")
    if not tt["stn"].isin(stops["stop_id"]).all():
        raise ValueError("stop_times contains an unknown stop_id")
    used_shapes = trips["shape_id"].dropna().astype(int)
    if not used_shapes.isin(shape_ids).all():
        raise ValueError("trips contains an unknown shape_id")


def validate_zip(path: Path, expected_headers: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as feed:
        if set(feed.namelist()) != set(expected_headers):
            raise ValueError(f"unexpected GTFS members: {sorted(feed.namelist())}")
        bad_member = feed.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed for {bad_member}")
        for name, expected_header in expected_headers.items():
            with feed.open(name) as member:
                header = member.readline().decode("utf-8").rstrip("\r\n")
            if header != expected_header:
                raise ValueError(f"unexpected header in {name}: {header}")


#%% Build feed
def build_feed(output_path: Path = OUTPUT) -> dict[str, int | str]:
    C.log("Loading prepared intercity tables")
    stns = C.load("ic-stations")
    lines = (
        C.load("ic-lines")
        .drop(columns="agency")
        .rename(columns={"operator": "agency"})
    )
    journeys = C.load("ic-journeys")
    datesets = C.load("ic-datesets")
    tt = C.load("ic-timetable")
    segments = C.load("ic-segments")
    tables = [stns, lines, journeys, datesets, tt, segments]
    if any(table is None for table in tables):
        raise FileNotFoundError("one or more prepared intercity tables are absent")
    validate_source_tables(stns, lines, journeys, datesets, tt, segments)

    C.log("Preparing GTFS agencies, stops, routes, trips and services")
    agencies, routes, lines = prepare_agencies_lines(stns, lines)
    stops = prepare_stops(stns)
    trips, services, starts, counts, journey_lines = prepare_trips_services(
        journeys, tt
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)

    expected_headers = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone",
        "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type",
        "routes.txt": (
            "route_id,agency_id,route_short_name,route_long_name,route_type"
        ),
        "trips.txt": "route_id,service_id,trip_id,shape_id",
        "stop_times.txt": (
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence,"
            "shape_dist_traveled"
        ),
        "calendar_dates.txt": "service_id,date,exception_type",
        "shapes.txt": (
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,"
            "shape_dist_traveled"
        ),
        "feed_info.txt": (
            "feed_publisher_name,feed_publisher_url,feed_lang,"
            "feed_start_date,feed_end_date,feed_version,feed_contact_url"
        ),
    }

    C.log(f"Writing GTFS archive to {output_path}")
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as feed:
            C.log("Writing shapes")
            line_shape_ids, line_stop_distances, shape_stats = write_shapes(
                feed, lines, segments, stns
            )
            line_shape = pd.Series(line_shape_ids, index=lines["line"])
            trip_shapes = line_shape.reindex(trips["route_id"]).to_numpy()
            trips["shape_id"] = pd.Series(
                np.where(trip_shapes >= 0, trip_shapes, pd.NA), dtype="Int32"
            )

            C.log("Preparing and writing stop times")
            sequence, arrivals, departures, shape_distances = (
                build_stop_time_arrays(
                    tt, lines, starts, counts, journey_lines,
                    line_stop_distances,
                )
            )
            validate_relations(
                agencies, stops, routes, trips, tt,
                np.arange(1, shape_stats["shapes"] + 1),
            )
            write_frame(feed, "agency.txt", agencies)
            write_frame(feed, "stops.txt", stops)
            write_frame(feed, "routes.txt", routes)
            write_frame(feed, "trips.txt", trips)
            write_stop_times(
                feed, tt, sequence, arrivals, departures, shape_distances
            )

            C.log("Writing service calendars")
            calendar_rows, start_date, end_date = write_calendar_dates(
                feed, datesets, services
            )
            feed_info = pd.DataFrame({
                "feed_publisher_name": ["TU Delft 3MARS"],
                "feed_publisher_url": ["https://www.tudelft.nl/"],
                "feed_lang": ["en"],
                "feed_start_date": [start_date],
                "feed_end_date": [end_date],
                "feed_version": [date.today().isoformat()],
                "feed_contact_url": ["https://www.tudelft.nl/"],
            })
            write_frame(feed, "feed_info.txt", feed_info)

        validate_zip(temporary_path, expected_headers)
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    stats = {
        "agencies": len(agencies),
        "stops": len(stops),
        "routes": len(routes),
        "trips": len(trips),
        "stop_times": len(tt),
        "services": len(services),
        "calendar_dates": calendar_rows,
        "previous_day_trips": int((tt["arr"].to_numpy()[starts] < 0).sum()),
        "feed_start_date": start_date,
        "feed_end_date": end_date,
        "archive_bytes": output_path.stat().st_size,
        "output": str(output_path),
    } | shape_stats
    return stats


#%% Run
if __name__ == "__main__":
    summary = build_feed()
    C.log("GTFS feed built and internally validated")
    for key, value in summary.items():
        C.log(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
