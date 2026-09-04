library(UK2GTFS)

# Load project configuration
env <- yaml::read_yaml("env.yml")

if (is.null(env$DATA_DIR)) {
  stop("DATA_DIR is missing from env.yml")
}

DATA <- normalizePath(
  path.expand(env$DATA_DIR),
  mustWork = TRUE
)

path_in <- file.path(DATA, "gtfs", "uk-atoc.zip")
out_dir <- file.path(DATA, "gtfs", "feeds")
out_name <- "man-UK_rail"

if (!file.exists(path_in)) {
  stop("ATOC input not found: ", path_in)
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

gtfs <- atoc2gtfs(
  path_in = path_in,
  ncores = min(8L, parallelly::availableCores()),
  public_only = TRUE,
  transfers = FALSE
)
gtfs_write(
  gtfs,
  folder = out_dir,
  name = out_name
)
message(
  "Created: ",
  file.path(out_dir, paste0(out_name, ".zip"))
)