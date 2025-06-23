# create a new folder called data-raw
# move rocios .rda file to that one
# save this script there, edit as needed.

library(dplyr)
#load it
load("./data-raw/new_training_data.rda")

# #change the name of the object, modify columns, rename, reorder as needed
# if you want to move some columns to the first positions you can also use dplyr::relocate().

#new_training_data <- new_training_data %>% ...

# use_data to this object, will recreate and compress automatically.
usethis::use_data_raw()
usethis::use_data(new_training_data, overwrite = TRUE)
