
library(dplyr)
#load it
load("./data-raw/fl_sample_data.rda")

# use_data to this object, will recreate and compress automatically.
usethis::use_data(fl_sample_data, compress = "xz", overwrite = TRUE)
