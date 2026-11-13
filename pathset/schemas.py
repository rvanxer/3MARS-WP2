"""Data contract of the path-set module.

Every table is a parquet file in ``PATHSET_DIR`` (``pathset.io.save``) unless a
``.npy`` file is named. Column order and dtypes below are binding: the modules
are written against them. Minutes are integers unless stated; clocks are the
LOCAL wall clock of the hub concerned.

Notation follows the deck (RA/Nov/path enumeration, slides 228-229): a path is
p = [e0, e1 ... en, en+1] with e0 the car access, e1..en service links and en+1
the car egress; mu is its mode sequence over A (air), B (bus/coach), R (rail);
transfers = n - 1.

ps-hubs                 one row per hub; station hubs first, then airports
    hub         int32   0..H-1
    kind        str     'S' station hub | 'A' airport
    name        str
    fua         int16   FUA id of a station hub; -1 for airports
    icc         str     country code
    tz_std      int8    standard-time UTC offset of the country, hours (0, 1, 2)
    lon, lat    float64
    iata        str     airports; '' for stations
    stns        list<int32>   member station ids of data-upstream-hubs; [] for airports
    released_id str     matching node id of the released graph ('' if none)
    match_m     float32 distance to that node, metres (NaN if none)

ps-links                one row per service link; sorted by link
    link        int32   0..L-1
    src, trg    int32   hubs
    mode        str     'A' | 'B' | 'R'
    cls         int16   riding-time class within (src, trg, mode), 0 = fastest
    ride        float32 riding time, minutes: mean over events weighted by dates
    ride_min, ride_max  int16
    tz_shift    int16   60 * (tz_std[trg] - tz_std[src]); arrival clock at trg =
                        departure clock at src + ride + tz_shift
    km          float32 great-circle distance src-trg
    ev0, ev1    int32   rows [ev0, ev1) of ps-events belong to this link
    n_events    int32
    active_dates int32  dates with at least one departure
    first_day, last_day int32   timeline day indices (pathset.dates)
    freq        float32 median over active dates of departures that day
    freq_max    int16   largest number of departures on one date
    freq_span   float32 departures / days from first_day to last_day
    freq_fixed  float32 departures inside FIXED_FREQ_START..END / days of that period
    active_share float32 active_dates / days from first_day to last_day

ps-sparse               links active on fewer than MIN_ACTIVE_DATES dates; same
                        columns as ps-links without ev0, ev1; not in the network

ps-events               one row per distinct run of a link; sorted by (link, dep, ride)
    link        int32
    dep         int16   local clock minute of departure at src, 0..1439
    ride        int16   elapsed minutes to arrival at trg (> 0)
    trip        int32   trip key: equal for all legs of one vehicle run pattern
    n_dates     int32   number of departure dates
ps-event-masks.npy      uint64 (n_events, N_WORDS): calendar of DEPARTURE dates
                        (local date at src), row-aligned with ps-events

ps-link-members         which operators a link pools
    link        int32
    agency      str     agency name (ground) or carrier code (air)
    route_type  int16   GTFS route type; -1 for air
    line        int64   line id of data-upstream-hubs; -1 for air
    n_events    int32

ps-drops                everything removed on the way, by rule
    step        str
    reason      str
    n           int64
    detail      str

ps-car                  direct car alternative
    o, d        int16   FUA ids
    minutes     float32
    km          float32
    fixed       bool    True when a cross-water crossing time was restored

ps-connectors           car access (e0) and egress (en+1) between a FUA and a hub
    fua         int16
    hub         int32
    acc_min     float32 FUA -> hub; NaN for an airport when above ACCESS_MAX_AIR_MIN
    egr_min     float32 hub -> FUA; NaN for an airport when above ACCESS_MAX_AIR_MIN
    source      str     'released' | 'fitted'

ps-moves                directed move between two different station hubs of one
                        FUA, or between an airport and a station hub of one of
                        its catchment FUAs (which need not be the airport's
                        city); a transfer's cities are those of both hubs
    hub_a, hub_b int32
    minutes     float32
    kind        str     'interhub' | 'fitted' | 'colocated' | 'road'

ps-transfers            one row per admissible ordered pair of links; sorted by link_in
    link_in, link_out  int32
    move        float32 minutes between trg(link_in) and src(link_out); 0 in one hub
    basis       str     'D' counted date by date | 'T' on typical days
    n_days      int32   dates (or weekdays) with at least one opportunity
    freq        float32 median over those days of opportunities that day
    ft_mean     float32 f_trans: mean arrival-to-departure minutes over opportunities
    ft_min      float32 smallest arrival-to-departure minutes

Windows (parameter WINDOWS) are keyed by the two modes: 'AA', 'AP' (air to bus
or rail), 'PA', 'PP'. A transfer from an arrival at clock a to a departure at
clock d is feasible when  lo + move <= d - a <= hi  and the two runs do not
have the same trip key.

ps-transfers keeps every admissible pair with at least one opportunity
(n_days >= 1); whether a chain EXISTS is decided by the consumer: basis 'D'
needs n_days >= MIN_ACTIVE_DATES, basis 'T' needs n_days >= 1.

ps-ftrans               f_trans at the key of the released 3MG graph: the
                        ps-transfers rows whose four hubs all have a released
                        id, mapped to those ids and pooled over riding-time
                        classes (operators listed, not split); the key of the
                        first six columns is unique
    in_src, in_trg, out_src, out_trg  str  released node ids
    mode_in, mode_out  str
    ft_mean     float32 mean of the rows' ft_mean weighted by freq x n_days /
                        days counted (the two links' common active dates on
                        basis 'D', 7 on basis 'T'), so that both bases weigh alike
    ft_min      float32
    n_rows      int32   ps-transfers rows pooled
    share_T     float32 share of those rows counted on typical days
    in_ops, out_ops  str  operators of the links, ';'-joined

--- Paths (pathset-enumerate.py). Shards per origin in PATHSET_DIR/paths/<table>/
    o<ooo>.parquet; the combined tables are ps-<table>.parquet. ---------------

Columns shared by the path tables (PATH):
    o, d        int16   origin and destination FUA
    seq         str     mode sequence over A, B, R; 'C' = direct car
    n           int8    transfers (links - 1; 0 for car)
    h_first, h_last int32  first boarding and last alighting hub (-1 car)
    l1, l2, l3  int32   service links (-1 unused); hubs of link k are its src, trg
    cities      str     route pattern: city ids, airports as IATA, '>'-joined
    key         str     link ids '-'-joined ('car' for car)
    acc, egr    float32 car access (e0) and egress (en+1) minutes
    mv1, mv2    float32 move minutes inside transfer 1, 2 (0 in one hub; NaN unused)
    ft1, ft2    float32 transfer minutes f_trans (NaN unused)
    t_a, t_dep, t_ivt, t_trans, t_tp, t_arr, t_e  float32  deck's GTT terms
    T           float32 clock minutes door to door
    gtt         float32 deck's GTT
    hwt         float32 hidden wait at the path frequency
    hwt_first   float32 hidden wait at the first link's frequency (deck's form)
    gtc         float32 gtt + hwt
    freq        float32 path frequency per day, capped by the sparsest link's freq_max
    freq_counted float32 path frequency as counted, before the cap
    basis       str     'D' | 'T'; '' for car
    n_days      int32   dates (weekdays on 'T') with an opportunity

ps-alternatives         kept alternatives; sorted by (o, d, rank)
    PATH + rank int16 (1.. in increasing gtc), cls str (mode class),
    freq_pooled float32, n_variants int32, status str (kept | kept_guarantee)
ps-variants             other members of each alternative, at most VARIANTS_KEPT
    o, d, rank (of the alternative), member int16 (1.. in increasing T, gtt,
    key), then PATH without o, d
ps-best-by-sequence     per (o, d, seq) the R0 candidate with the least gtt
                        (before R3); the input of the CASPT accessibility curves
    PATH
ps-funnel               per FUA pair
    o, d; n_r0, n_r0_bound, n_r3, n_r1, n_r2, n_kept, n_capped int64;
    n_classes int8; best_T, best_gtt, car_min float32; pt bool (a service
    alternative is kept). n_r0_bound = three-link R0 candidates counted
    without evaluating their existence because the bound cut them, so that
    n_r0 - n_r0_bound <= exact R0 count <= n_r0 (0 at MU = inf)
ps-seq-counts           per origin and mode sequence, summed over destinations
    o int16, seq str, n_r0, n_r0_bound, n_r3, n_kept int64
ps-unserved             pairs without a service alternative
    o, d int16; reason str ('no_service_path' | 'nothing')
ps-trace                named pairs only: every R0 candidate with its fate
    PATH + cand int64, cls, alt int64, status, rule, by int64, detail
    (status, rule, by, detail as rules.TRACE)
"""

MODES = ("A", "B", "R")

HUBS = ["hub", "kind", "name", "fua", "icc", "tz_std", "lon", "lat", "iata",
        "stns", "released_id", "match_m"]
LINKS = ["link", "src", "trg", "mode", "cls", "ride", "ride_min", "ride_max",
         "tz_shift", "km", "ev0", "ev1", "n_events", "active_dates", "first_day",
         "last_day", "freq", "freq_max", "freq_span", "freq_fixed", "active_share"]
EVENTS = ["link", "dep", "ride", "trip", "n_dates"]
LINK_MEMBERS = ["link", "agency", "route_type", "line", "n_events"]
DROPS = ["step", "reason", "n", "detail"]
CAR = ["o", "d", "minutes", "km", "fixed"]
CONNECTORS = ["fua", "hub", "acc_min", "egr_min", "source"]
MOVES = ["hub_a", "hub_b", "minutes", "kind"]
TRANSFERS = ["link_in", "link_out", "move", "basis", "n_days", "freq",
             "ft_mean", "ft_min"]
FTRANS = ["in_src", "in_trg", "out_src", "out_trg", "mode_in", "mode_out",
          "ft_mean", "ft_min", "n_rows", "share_T", "in_ops", "out_ops"]

PATH = ["o", "d", "seq", "n", "h_first", "h_last", "l1", "l2", "l3", "cities",
        "key", "acc", "egr", "mv1", "mv2", "ft1", "ft2", "t_a", "t_dep",
        "t_ivt", "t_trans", "t_tp", "t_arr", "t_e", "T", "gtt", "hwt",
        "hwt_first", "gtc", "freq", "freq_counted", "basis", "n_days"]
ALTERNATIVES = PATH + ["rank", "cls", "freq_pooled", "n_variants", "status"]
VARIANTS = ["o", "d", "rank", "member"] + PATH[2:]
BEST_BY_SEQUENCE = PATH
FUNNEL = ["o", "d", "n_r0", "n_r0_bound", "n_r3", "n_r1", "n_r2", "n_kept",
          "n_capped", "n_classes", "best_T", "best_gtt", "car_min", "pt"]
SEQ_COUNTS = ["o", "seq", "n_r0", "n_r0_bound", "n_r3", "n_kept"]
UNSERVED = ["o", "d", "reason"]
TRACE = PATH + ["cand", "cls", "alt", "status", "rule", "by", "detail"]


def window_key(mode_in: str, mode_out: str) -> str:
    """Key of the transfer window for an ordered pair of modes."""
    return ("A" if mode_in == "A" else "P") + ("A" if mode_out == "A" else "P")
