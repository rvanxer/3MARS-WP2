script_dir <- Sys.getenv("UK_RAIL_PROJECT_DIR", unset = "")
if (!nzchar(script_dir)) {
    file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (!length(file_arg)) {
        stop("Cannot locate the project directory. Run: bash uk-rail.sh")
    }
    script_file <- sub("^--file=", "", file_arg[[1]])
    script_dir <- dirname(normalizePath(script_file, mustWork = TRUE))
}
script_dir <- normalizePath(script_dir, mustWork = TRUE)

# Load project configuration
env_file <- file.path(script_dir, "env.yml")
if (!file.exists(env_file)) {
    stop("Configuration file not found: ", env_file)
}
env <- yaml::read_yaml(env_file)

if (
    is.null(env$DATA_DIR) ||
    !is.character(env$DATA_DIR) ||
    length(env$DATA_DIR) != 1L ||
    !nzchar(env$DATA_DIR)
) {
    stop("DATA_DIR must be a non-empty path in env.yml")
}

DATA <- normalizePath(
    path.expand(env$DATA_DIR),
    mustWork = TRUE
)

path_in <- file.path(DATA, "gtfs", "uk-atoc.zip")
out_dir <- file.path(DATA, "gtfs", "feeds")
out_name <- "ext-UK_rail"
out_file <- file.path(out_dir, paste0(out_name, ".zip"))

if (file.exists(out_file)) {
    message("Output already exists; skipping: ", out_file)
    quit(save = "no", status = 0)
}

if (!file.exists(path_in)) {
    stop("ATOC input not found: ", path_in)
}

required_packages <- c("UK2GTFS", "parallelly", "yaml")
missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
    stop(
        "Missing R packages: ", paste(missing_packages, collapse = ", "),
        ". Run: bash uk-rail.sh"
    )
}

expected_r_version <- "4.5.2"
if (as.character(getRversion()) != expected_r_version) {
    stop(
        "R ", expected_r_version, " is required; found R ", getRversion(),
        ". Run: bash uk-rail.sh"
    )
}

expected_uk2gtfs_sha <- "87d0545a38f040be7ada5d7088f3169dd3da7e9b"
uk2gtfs_description <- utils::packageDescription("UK2GTFS")
if (!identical(uk2gtfs_description$RemoteSha, expected_uk2gtfs_sha)) {
    stop("The installed UK2GTFS revision is incorrect. Run: bash uk-rail.sh")
}
suppressPackageStartupMessages(library(UK2GTFS))

# UK2GTFS currently resolves activity_codes through its attached package data.
invisible(UK2GTFS:::clean_activities2("TB", public_only = TRUE))

message(
    "Using UK2GTFS ", as.character(utils::packageVersion("UK2GTFS")),
    " from ", find.package("UK2GTFS")
)

if (identical(Sys.getenv("UK_RAIL_CHECK_ONLY"), "1")) {
    message("Preflight checks passed; conversion was not run.")
} else {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    gtfs <- UK2GTFS::atoc2gtfs(
        path_in = path_in,
        ncores = min(8L, parallelly::availableCores()),
        public_only = TRUE,
        transfers = FALSE
    )
    UK2GTFS::gtfs_write(
        gtfs,
        folder = out_dir,
        name = out_name
    )
    message(
        "Created: ",
        file.path(out_dir, paste0(out_name, ".zip"))
    )
}
