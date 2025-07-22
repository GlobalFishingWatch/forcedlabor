
library(dplyr)
#load it
load("./data-raw/fl_training.rda")

# #change the name of the object, modify columns, rename, reorder as needed
# if you want to move some columns to the first positions you can also use dplyr::relocate().

fl_training <- new_training_data %>%
  select(known_offender, known_non_offender, indID, source_id_number, gear,
         engine_power_kw, tonnage_gt, length_m, position_messages,
         hours, fishing_hours, average_daily_fishing_hours,
         fishing_hours_foreign_eez, fishing_hours_high_seas,
         max_distance_from_shore_km, max_distance_from_port_km,
         number_encounters, number_forced_labor_encounters,
         average_encounter_duration_hours, gaps_12_hours,
         average_off_distance_from_port_km, average_off_distance_from_shore_km,
         average_gap_days, average_gap_km, number_foreign_port_visits,
         number_loitering_events, average_loitering_duration_hours,
         average_voyage_duration_hours, number_voyages, flag_region)


# use_data to this object, will recreate and compress automatically.
usethis::use_data(fl_training, overwrite = TRUE)

