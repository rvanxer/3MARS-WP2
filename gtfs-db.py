"""Clean original GTFS feeds into a database for downstream use."""

#%% Imports
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile

I16, I32 = np.int16, np.int32

import config as C

params = C.load_params()

#%% Compile feed info and assign feed_id
feeds = []
imp_tables = ("agency", "routes", "stops", "trips", "stop_times",
              "calendar", "calendar_dates")
for file in sorted(C.DATA.glob("gtfs/feeds/*.zip")):
    with ZipFile(file, "r") as zf:
        size_zip = sum([f.compress_size for f in zf.infolist()]) / 1024 ** 2
        size_unzip = sum([f.file_size for f in zf.infolist()]) / 1024 ** 2
        for f in zf.namelist():
            table = f.split("/")[-1].removesuffix(".txt")
            if table in imp_tables:
                feeds.append({
                    "name": file.stem,
                    "size_zip": size_zip,
                    "size_unzip": size_unzip,
                    "table": table
                })
feeds = (
    pd.DataFrame(feeds)
    .assign(_=1)
    .pivot_table("_", ["name", "size_zip", "size_unzip"], "table")
    .fillna(0).astype(bool)
    .rename(columns=lambda x: "f_" + x)
    .rename_axis(None, axis=1)
    .reset_index()
    .rename_axis("id")
    .reset_index()
    .astype({"id": I16})
)#.view()
C.save(feeds, "feed-info")

#%% Read generic GTFS table
def read_gtfs_table(
    feed: str,
    table: str,
    root: str | Path = C.DATA / "gtfs/feeds",
    feeds: pd.DataFrame = feeds,
    cols: list[str] = [],
    dtypes: dict[str] = None,
    warn: bool = False
):
    """Load a table from a zipped GTFS feed file."""
    try:
        feed_id = I16(feeds[feeds["name"] == feed]["id"].iloc[0])
        with ZipFile(Path(root) / f"{feed}.zip", "r") as zf:
            files = [f for f in zf.namelist() if Path(f).stem == table]
            if len(files) == 0:
                df = pd.DataFrame([], columns=cols)
                df.insert(0, "fid", feed_id)
                return df
            # first identify the available columns in the file
            with zf.open(files[0]) as f:
                file_cols = list(pd.read_csv(f, nrows=0).columns)
                # filter the available required columns
                if len(cols) > 0:
                    file_cols = [c for c in file_cols if c.strip() in cols]
            with zf.open(files[0]) as f:
                df = pd.read_csv(f, usecols=file_cols,
                                 dtype=dtypes, low_memory=False)
                df.rename(columns=str.strip, inplace=True)
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
        df = df if len(cols) == 0 else df[cols]
        df.insert(0, "fid", I16(feed_id))
    except Exception as e:
        if warn:
            C.error(f"{feed}: {e}")
        df = pd.DataFrame([], columns=["fid"] + list(cols))
    return df

# df = read_gtfs_table("man-Estonia", "stops"); df

#%% Stops 
def get_stops(feed):
    df = read_gtfs_table(feed, "stops", cols=[
        "stop_id", "stop_name", "stop_lon", "stop_lat"])
    df.columns = ["fid", "stop_id", "name", "lon", "lat"]
    df = df.astype({"stop_id": str, "lon": np.float32, "lat": np.float32})
    if len(df) == 0:
        # C.warn(f"{feed}: Empty stops table")
        return
    return df.rename_axis("id")
    
# get_stops("man-Trenitalia")

#%% Routes & agencies
def get_routes(feed):
    agency = (
        read_gtfs_table(feed, "agency", cols=[
            "agency_id", "agency_name", "agency_timezone"])
        .rename(columns={"agency_name": "agency", "agency_timezone": "tz"})
        .astype({"agency_id": str, "tz": str})
    )
    df = read_gtfs_table(
        feed, "routes", cols=["route_id", "agency_id", "route_short_name",
                              "route_long_name", "route_type"]
    )
    df["mode_id"] = df.pop("route_type").astype(str)
    df["name"] = (
        df["route_short_name"].astype(str).fillna("") + " | " +
        df["route_long_name"].astype(str).fillna("")
    ).str.strip(" | ")
    return (
        df[df["mode_id"].str.isdigit()]
        .astype({"agency_id": str})
        .merge(agency[["agency_id", "agency", "tz"]], "left", on="agency_id")
        .astype({"route_id": str, "mode_id": I16, "agency_id": str})
        [["fid", "route_id", "name", "agency", "tz", "mode_id"]]
        .rename_axis("id")
    )

# get_routes("man-Estonia")

#%% Service dates
def get_service_dates(feed: str,
                      start_date: datetime.date = params.BASE_START_DATE,
                      end_date: datetime.date = params.BASE_END_DATE):
    cal, removed = [pd.DataFrame([], columns=["service_id", "date"])] * 2
    start_int = int(str(start_date).replace("-", ""))
    ## Calendar table
    df = read_gtfs_table(feed, "calendar")
    if len(df) > 0:
        days = df.loc[:, "monday": "sunday"].apply(tuple, axis=1)
        df = (pd.concat([df[["service_id", "start_date", "end_date"]],
                        days.rename("days")], axis=1)
              .groupby(["start_date", "end_date", "days"])
              ["service_id"].agg(list).reset_index())
        for col in ["start_date", "end_date"]:
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d")
        def get_imp_dates(r):
            dates = pd.date_range(r["start_date"], r["end_date"])
            d = dates[dates >= pd.to_datetime(start_date)]
            d = dates[dates <= pd.to_datetime(end_date)]
            d = d[d.day_of_week.map(dict(enumerate(r["days"]))).astype(bool)]
            return d.year * 10_000 + d.month * 100 + d.day
        df["date"] = df.apply(get_imp_dates, axis=1)
        df = df[["service_id", "date"]].explode("date")
        df = df[df["date"] >= start_int]
        df = df.explode("service_id")
        cal = pd.concat([cal, df])
    ## Calendar dates table
    df = read_gtfs_table(feed, "calendar_dates")
    if len(df) > 0:
        df = df[df["date"] >= start_int]
        include = df[df["exception_type"] == 1][["service_id", "date"]]
        exclude = df[df["exception_type"] == 2][["service_id", "date"]]
        cal = pd.concat([cal, include])
        removed = pd.concat([removed, exclude])
    ## Combine the two tables
    df = pd.concat([cal, removed])
    df["day_id"] = I32(df["date"] - start_int)
    df.sort_values("day_id", inplace=True)
    df.drop_duplicates(keep=False, inplace=True)
    df = df.astype({"service_id": str})
    df = df.groupby("service_id")["day_id"].agg(tuple).reset_index()
    df = df.groupby("day_id")["service_id"].agg(list).reset_index()
    return df.rename_axis("id")

# x = get_service_dates("man-Estonia"); x

#%% Process feed
def process_feed(feed, overwrite=False):
    outdir = C.mkdir(C.DATA / "gtfs/clean") / feed
    if outdir.exists() and not overwrite:
        return
    ## Stops, routes and service dates
    stops = get_stops(feed)
    routes = get_routes(feed)
    dates = get_service_dates(feed)
    ## Trips
    trips = (
        read_gtfs_table(feed, "trips", cols=["trip_id", "route_id", "service_id"])
        .astype({"trip_id": str, "route_id": str, "service_id": str})
        .merge(routes[["route_id"]].reset_index(), on="route_id")
        .drop(columns="route_id")
        .rename(columns={"id": "route_id"})
        .merge(dates["service_id"].explode().reset_index(), on="service_id")
        .rename(columns={"id": "dateset_id"})
        .astype({"route_id": I32, "dateset_id": I32})
        [["fid", "trip_id", "route_id", "dateset_id"]]
        .rename_axis("id")
    )
    if len(trips) == 0:
        # C.warn(f"{feed}: Empty trips table")
        return
    ## Timetable (stop times)
    tt = read_gtfs_table(
        feed, "stop_times",
        cols=["trip_id", "stop_sequence", "stop_id",
              "arrival_time", "departure_time"]
    ).astype({"stop_id": str, "trip_id": str})
    tt = tt.merge(stops[["stop_id"]].reset_index(), on="stop_id")
    tt = tt.drop(columns="stop_id").rename(columns={"id": "stop_id"})
    tt = tt.merge(trips[["trip_id"]].reset_index(), on="trip_id")
    tt = tt.drop(columns="trip_id").rename(columns={"id": "trip_id"})
    tt = tt.astype({"stop_id": I32, "trip_id": I32})
    tt.rename(columns={"stop_sequence": "snum", "arrival_time": "arr_time",
                       "departure_time": "dep_time"}, inplace=True)
    tt = tt[['trip_id', 'stop_id', 'snum', 'arr_time', 'dep_time']]
    tt = tt[tt["snum"] <= np.iinfo(np.uint16).max]
    if len(tt) == 0:
        # C.warn(f"{feed}: Empty stop times table")
        return
    for col in ["arr_time", "dep_time"]:
        vals = tt[col].astype("category")
        cats = vals.cat.categories.astype(str)
        cats = cats[cats.str.strip() != ""]
        h, m, s = list(zip(*cats.str.split(":")))
        time = I32(h) * 3600 + I32(m) * 60 + I32(s)
        tt[col] = vals.map(dict(zip(cats, time)))
    tt = tt.dropna(subset=["arr_time", "dep_time"])
    tt = tt.astype({"arr_time": I32})
    tt["wait"] = I16(tt.pop("dep_time") - tt["arr_time"])
    ## Trips as stop & time sequences
    trips2 = (
        tt.sort_values("snum", ignore_index=True)
        .groupby("trip_id", sort=False)
        .agg({"stop_id": tuple, "arr_time": list, "wait": list})
    )
    trips2["start"] = I32(trips2["arr_time"].str[0])
    trips2["end"] = I32(trips2["arr_time"].str[-1])
    relative_diff = lambda r: tuple(np.array(r["arr_time"]) - r["start"])
    trips2["arr_time"] = trips2.apply(relative_diff, axis=1)
    ## Stop sequences
    stop_seq = (
        trips2["stop_id"].reset_index()
        .groupby("stop_id")["trip_id"].agg(list)
        .reset_index().rename_axis("id")
    )
    ## Time sequences
    time_seq = (
        trips2.reset_index()
        .groupby("arr_time")
        .agg({"trip_id": list, "wait": "first"})
        .reset_index().rename_axis("id")
    )
    ## Update condensed trips table
    trips2 = (
        trips2.merge(stop_seq["trip_id"].explode().reset_index(), on="trip_id")
        .drop(columns="stop_id")
        .astype({"id": I32})
        .rename(columns={"id": "stopseq_id"})
    )
    trips2 = (
        trips2.merge(time_seq["arr_time"].reset_index(), on="arr_time")
        .drop(columns=["arr_time", "wait"])
        .astype({"id": I32})
        .rename(columns={"id": "timeseq_id", "trip_id": "id"})
        .merge(trips, on="id")
        .set_index("id")
        .sort_index().reset_index()
        .rename(columns={"trip_id": "orig_id"})
        [["id", "orig_id", "route_id", "dateset_id",
          "stopseq_id", "timeseq_id", "start", "end"]]
    )
    ## Update other tables: routes, stops, dates, stop & time sequences
    routes = (
        routes.reset_index()
        .astype({"id": I32, "agency": str})
        .rename(columns={"route_id": "orig_id"})
        .merge(trips2["route_id"].rename("id").drop_duplicates(), on="id")
    )
    stops = (
        stops.reset_index()
        .astype({"id": I32})
        .rename(columns={"stop_id": "orig_id"})
        [["id", "orig_id", "name", "lon", "lat"]]
    )
    stop_seq = stop_seq[["stop_id"]].reset_index().astype({"id": I32})
    time_seq = time_seq[["arr_time", "wait"]].reset_index().astype({"id": I32})
    dates = dates[["day_id"]].reset_index().astype({"id": I32})
    ## Export tables
    for table, df in {
        "stops": stops,
        "stop_seq": stop_seq,
        "routes": routes,
        "trips": trips2,
        "datesets": dates,
        "time_seq": time_seq
    }.items():
        df.to_parquet(C.mkdir(outdir) / f"{table}.parquet")

# x = process_feed("man-Trenitalia", overwrite=True); x
# x = process_feed("man-Finland"); x # 33s

#%% Process all feeds [≈33:00]
C.log("Cleaning original feeds")
for feed in (pbar := tqdm(feeds.name)):
    pbar.set_description(feed)
    try:
        process_feed(feed, overwrite=False)
    except Exception as e:
        C.error(f"{feed}: {str(e).split('\n')[0]}")
        pass

#%% Combine all tables [0:19]
feed2id = feeds.set_index("name")["id"].astype(I16)
for table in [
    "datesets",
    "routes",
    "stop_seq",
    "stops",
    "time_seq",
    "trips"
]:
    df_all = []
    for file in C.DATA.glob(f"gtfs/clean/*/{table}.parquet"):
        df = pd.read_parquet(file)
        df.drop(columns=["fid", "orig_id"], errors="ignore", inplace=True)
        df.insert(0, "fid", feed2id.loc[file.parent.stem])
        df_all.append(df)
    df = pd.concat(df_all, ignore_index=True)
    C.save(df, f"gtfs/db-{table}")
