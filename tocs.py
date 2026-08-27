"""Identify major transport operating companies (TOCs) from intercity 
GTFS agencies."""

#%% Imports
import pandas as pd

import config as C

#%% Load data
fuas = C.load("fuas").set_index("id").rename_axis("fua")#.view()
stns = C.load("ic-stations").set_index("stn")#.view()
lines = C.load("ic-lines").set_index("line")#.view()
seg = C.load("ic-segments")[["src", "trg", "line"]]#.view()

#%% Agencies having intercity lines
ic_agency = (
    lines.explode("stn").astype({"stn": int}).reset_index()
    .merge(stns[["fua"]], on="stn")
    [["line", "agency", "rail", "fua"]].drop_duplicates()
    .groupby(["line", "agency", "rail"])
    ["fua"].agg(lambda x: tuple(sorted(list(x)))).reset_index()
    .groupby(["agency", "rail", "fua"])
    ["line"].agg(list).reset_index()
    .pipe(lambda df: df[df["fua"].apply(len) > 1])
    .assign(od=lambda df: [list(zip(x[:-1], x[1:])) for x in df["fua"]])
    .explode("od", ignore_index=True)
    .groupby(["rail", "agency"])["od"].agg(["count", list])
    .set_axis(["n_od", "od"], axis=1).reset_index()
    .sort_values("n_od", ascending=False, ignore_index=True)
)#.view()

#%% Important agencies with covered countries
imp_agencies = []
for mode, is_rail in [("Bus", False), ("Rail", True)]:
    od_by_agency = (
        ic_agency.query(f"rail == {is_rail}")
        [["agency", "od"]].explode("od").assign(_=True)
        .pivot_table("_", "agency", "od", sort=False)
        .fillna(0).astype(bool)
    )
    ods = set()
    df = []
    for agency, r in od_by_agency.iterrows():
        ods.update(new := (set(r[r].index) - ods))
        df.append({"agency": agency, "n_ods": len(new)})
    df = pd.DataFrame(df).sort_values("n_ods", ascending=0)
    df = df.query("n_ods > 0").assign(mode=mode)
    imp_agencies.append(df[["agency", "mode", "n_ods"]])
imp_agencies = (
    pd.concat(imp_agencies)
    .sort_values("n_ods", ascending=False, ignore_index=True)
)
## Codes of countries covered by agencies
seg["src_icc"] = seg["src"].map(stns["fua"].map(fuas["icc"]))
seg["trg_icc"] = seg["trg"].map(stns["fua"].map(fuas["icc"]))
icc = (
    seg.explode("line")
    .merge(lines[["agency", "rail"]], on="line")
    [["agency", "rail", "src_icc", "trg_icc"]]
    .melt(["agency", "rail"], value_name="icc")
    .groupby(["agency", "rail"])
    ["icc"].agg(lambda x: ",".join(sorted(set(x))[:5]))
    .rename("country").reset_index()
)
icc["mode"] = icc.pop("rail").map({False: "Bus", True: "Rail"})
imp_agencies = imp_agencies.merge(icc, on=["agency", "mode"])#.view()

#%% Export major agencies
imp_agencies.to_csv(C.DATA / "gtfs/imp-agencies.csv", index=False)

#%% Manually map agency to operator
agency2toc = (
    pd.read_excel(C.DATA / "gtfs/agency2toc.xlsx")
    .assign(rail=lambda df: df["mode"] == "Rail")
    [["agency", "rail", "operator", "domicile"]]
)#.view(5)
print("Total operators:", agency2toc["operator"].nunique())

#%% Intercity lines, stations and associated data
ic_lines = (
    lines.query("intercity").reset_index()
    .merge(agency2toc, on=["agency", "rail"])
)#.view()
major_stns = ic_lines["stn"].explode().astype(int).unique()
stns2 = stns.loc[major_stns].sort_index().reset_index()#.view()
ic_jrn = (
    C.load("ic-journeys")
    .merge(ic_lines[["line"]], on="line")
)#.view()
ic_dates = (
    C.load("ic-datesets")
    .merge(ic_jrn[["dateset"]].drop_duplicates(), on="dateset")
)#.view()
ic_tt = (
    C.load("ic-timetable")
    .merge(ic_jrn[["jrn"]], on="jrn")
)#.view()

#%% Intraurban data involving major stations on intercity lines
urb_lines = (
    lines.query("~intercity")
    ["stn"].explode().astype(int).reset_index()
    .merge(stns2[["stn"]], on="stn")
    .groupby("line")
    ["stn"].agg(["count", list])
    .query("count > 1")
    ["list"].rename("stn").reset_index()
    .merge(lines.drop(columns="stn"), on="line")
    .assign(operator=".Local")
)#.view()
urb_jrn = (
    C.load("ic-journeys")
    .merge(urb_lines[["line"]], on="line")
)#.view()
urb_dates = (
    C.load("ic-datesets")
    .merge(urb_jrn[["dateset"]].drop_duplicates(), on="dateset")
)#.view()
urb_tt = (
    C.load("ic-timetable")
    .merge(urb_jrn[["jrn", "line"]], on="jrn")
    .merge(stns2[["stn"]], on="stn")
)#.view()

#%% Combine intercity and intraurban data
lines2 = (
    pd.concat([ic_lines, urb_lines])
    .drop_duplicates("line")
    .sort_values("line", ignore_index=True)
    [["line", "agency", "operator", "rail", "tz_gap", "stn", "intercity"]]
)#.view()
jrn = (
    pd.concat([ic_jrn, urb_jrn])
    .drop_duplicates("jrn")
    .sort_values("jrn", ignore_index=True)
)#.view()
dates = (
    pd.concat([ic_dates, urb_dates])
    .drop_duplicates("dateset")
    .sort_values("dateset", ignore_index=True)
)#.view()
tt = (
    pd.concat([ic_tt, urb_tt]).drop(columns="line")
    .sort_values(["jrn", "arr"], ignore_index=True)
)#.view()
seg2 = (
    seg["line"].explode().astype(int).reset_index()
    .merge(lines2[["line"]], on="line")
    .groupby("seg")["line"].agg(list).reset_index()
    .merge(C.load("ic-segments").drop(columns="line"), on="seg")
    .merge(stns2["stn"].rename("src"), on="src")
    .merge(stns2["stn"].rename("trg"), on="trg")
    .set_index("seg")
    [["src", "trg", "mode", "intercity", "line"]]
)#.view()

#%% Export updated tables
C.log("Saving intercity data tables")
for df, table in [
    (stns2, "ic-stations"),
    (lines2, "ic-lines"),
    (jrn, "ic-journeys"),
    (dates, "ic-datesets"),
    (tt, "ic-timetable"),
    (seg2, "ic-segments"),
]:
    C.log(f"Table `{table}`: {len(df):,} rows")
    C.save(df, table)
