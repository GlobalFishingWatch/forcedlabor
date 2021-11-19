#' Gear manual correction
#' Will need updating with more offenders and non offenders data
#'
#' @param data data frame. Needs to have a gear (character) and ssvid (character) columns.
#' @return data frame with corrected gears
#'
#' @importFrom dplyr mutate
#'
#' @export
#'

gear_correction <- function(data){

  data %>%
    # manually correct some gears
    dplyr::mutate(gear = dplyr::if_else(gear == "fishing" &
                            ssvid %in% c("412440381", "412440436", "412440539",
                                         "412440432", "412440381", "412440512",
                                         "412440378"),
                          "purse_seines", gear)) %>%
    dplyr::mutate(gear = dplyr::if_else(gear == "fishing" &
                            ssvid %in% c("412209209", "250000216", "235004140"),
                          "trawlers", gear)) %>%
    dplyr::mutate(gear = dplyr::if_else(gear == "fishing" &
                            ssvid %in% c("412200225"),
                          "drifting_longlines", gear)) %>%
    dplyr::mutate(gear = dplyr::if_else(gear == "fishing" &
                            ssvid %in% c("412410028"),
                          "tuna_purse_seines", gear)) %>%
    dplyr::mutate(gear = dplyr::if_else(gear == "fishing" &
                            ssvid %in% c("412440494", "412440495", "412440499",
                                         "412440761"),
                          "squid_jiggers", gear))

}


#' Preprocessing of offenders, possible offenders, non offenders and unlabeled data
#'
#' @param fl_data data frame of offenders. Needs to have columns such as: a gear (character),
#' ssvid (character), engine_power_kw (double), tonnage_gt (double),
#' length_m (double), ais_type (character), event_ais_year (integer),
#' fl_event_id (integer), known_offender (double), known_non_offender (integer),
#' possible_offender (double)
#' @param tidy_data data frame of AIS. Needs to have the same columns as fl_data.
#' @param gears_interest vector of character elements with names of the gears of interest.
#' @param vars_to_factor vector of character elements with names of columns to convert
#' from character to factor. They have to be columns existing in fl_data and tidy_data.
#' @param vars_remove: vector of character elements with names of columns to remove from tidy_data.
#' @return list with 2 elements:
#' holdout_set : data frame with AIS info from offenders before/after
#' the year of offense, potential offenders and known non offenders;
#' training set : data frame with AIS info from offenders during the year
#' of offense, and unlabeled cases
#'
#' @import dplyr
#' @importFrom forcats fct_relevel
#'
#' @export
#'

ml_prep_data <- function(fl_data = fl_df$data,
                           tidy_data = alltogether_tidy$data ,
                           gears_interest, vars_to_factor, vars_remove){

  utils::globalVariables("where")

  # fl events with gear correction
  fl_gear <- fl_data %>%
    gear_correction() %>%
    # keep only gears of interest
    dplyr::filter(gear %in% gears_interest) %>%
    # dropping NAs in engine_power_kw, tonnage_gt, length_m and ais_type
    dplyr::filter(!is.na(engine_power_kw) & !is.na(tonnage_gt) & !is.na(length_m) & !is.na(ais_type))

  fl_na_zero <- fl_gear %>%
    dplyr::select(tidyselect:::where(~ is.numeric(.x) && any(is.na(.x)))) %>%
    apply(MARGIN = 2, function(x){ifelse(is.na(x), 0, x)})
  fl_gear[,colnames(fl_na_zero)] <- fl_na_zero


  # fl events from years previous or after the event
  fl_out_events <- fl_gear %>%
    dplyr::filter(event_ais_year == 0) %>%
    dplyr::mutate(fl_event_id = as.character(fl_event_id))

  # offenders from other years
  fl_out_offenders <- fl_out_events %>%
    dplyr::filter(known_offender == 1)

  # non offenders from any year
  fl_out_non_offenders <- fl_gear %>%
    dplyr::filter(known_non_offender == 1)

  # possible offenders any year
  fl_out_possible_offenders <- fl_gear %>%
    dplyr::filter(possible_offender == 1)


  # now only the tidy data
  tidy_gear <- tidy_data %>%
    gear_correction() %>%
    # keep only gears of interest
    dplyr::filter(gear %in% gears_interest) %>%
    # dropping NAs in engine_power_kw, tonnage_gt, length_m and ais_type
    dplyr::filter(!is.na(engine_power_kw) & !is.na(tonnage_gt) & !is.na(length_m) & !is.na(ais_type))

  tidy_na_zero <- tidy_gear %>%
    dplyr::select(tidyselect:::where(~ is.numeric(.x) && any(is.na(.x)))) %>%
    apply(MARGIN = 2, function(x){ifelse(is.na(x), 0, x)})

  tidy_gear[,colnames(tidy_na_zero)] <- tidy_na_zero

  # now only the unlabeled
  # this should have removed all of the known offenders, possible offenders and non offenders,
  # before, during and after the events
  unlabeled_df <- tidy_gear %>%
    dplyr::mutate(sum_fl = known_offender + known_non_offender + possible_offender) %>%
    dplyr::filter(sum_fl == 0) %>%
    dplyr::select(-sum_fl)

  # adding the offenders with AIS during the event
  offenders_event <- fl_gear %>%
    dplyr::filter(known_offender == 1 & event_ais_year == 1)

  unlabeled_offenders <- rbind.data.frame(offenders_event, unlabeled_df) %>%
    dplyr::mutate_if(is.logical, as.numeric) %>%
    dplyr::mutate(source_id = ifelse(known_offender != 1,
                              paste0("no_source_", row_number()),
                              source_id)) %>%
    dplyr::mutate(fl_event_id = ifelse(known_offender != 1,
                                paste0("no_fl_info_", row_number()),
                                fl_event_id)) %>%
    dplyr::mutate_if(is.logical, as.numeric) %>%
    dplyr::mutate_at(all_of(vars_to_factor), as.factor) %>%
    # Relevel to ensure model metrics are calculated properly
    dplyr::mutate(known_offender = fct_relevel(known_offender,c("1","0")))%>%
    # Everything needs to be numeric for DALEX
    dplyr::mutate(across(tidyselect:::where(is.integer),as.numeric)) %>%
    # adding vessel-year ID
    dplyr::mutate(indID = paste(ssvid,year,sep = "-"))

  # making a hold-out test set

  holdout_df <- rbind.data.frame(fl_out_offenders,
                                 fl_out_non_offenders,
                                 fl_out_possible_offenders) %>%
    dplyr::select(-all_of(vars_remove)) %>%
    dplyr::mutate_if(is.logical, as.numeric) %>%
    dplyr::mutate_at(all_of(vars_to_factor), as.factor)%>%
    # Relevel to ensure model metrics are calculated properly
    dplyr::mutate(known_offender = forcats::fct_relevel(known_offender,c("1","0")))%>%
    # Everything needs to be numeric for DALEX
    dplyr::mutate(across(tidyselect:::where(is.integer),as.numeric)) %>%
    # adding vessel-year ID
    dplyr::mutate(indID = paste(ssvid,year,sep = "-"))


  return(list(holdout_set = holdout_df,
              training_set = unlabeled_offenders))


}
