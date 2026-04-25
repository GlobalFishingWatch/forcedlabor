
library(dplyr)
#load it
load("./data-raw/sample_data.rda")

# use_data to this object, will recreate and compress automatically.
usethis::use_data(sample_data, compress = "xz", overwrite = TRUE)
