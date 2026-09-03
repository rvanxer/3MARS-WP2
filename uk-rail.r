install.packages("remotes")
remotes::install_github("ITSLeeds/UK2GTFS")
library(UK2GTFS) # 5s

path_in <- "~/Documents/research-data/3mars/gtfs/uk-atoc"
files <- list.files(path_in, full.names = TRUE)
# flf <- files[grepl("\\.flf$", files, ignore.case = TRUE)]
flf <- files[grepl(".flf", files, ignore.case = TRUE)]

gtfs <- atoc2gtfs(
  path_in = "~/Documents/research-data/3mars/gtfs/uk-atoc.zip",
  ncores = 8, public_only = TRUE, transfers = FALSE
) # 3m46s

out_zip <- "~/Downloads/uk_rail_gtfs.zip"
tmp <- tempfile("gtfs_")
dir.create(tmp)
# write each table to a .txt inside tmp
for (nm in names(gtfs)) {
  utils::write.table(gtfs[[nm]], quote = TRUE, na = "",
                     file = file.path(tmp, paste0(nm, ".txt")),
                     sep = ",", row.names = FALSE, col.names = TRUE)
} # 22s
setwd(tmp); zip::zipr(out_zip, list.files(".", full.names = FALSE))
