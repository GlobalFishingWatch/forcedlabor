#' Preprocessing of offenders, non offenders and unlabeled
#' data
#'
#' @param tidy_data data frame of AIS. Needs to have columns such as: a gear
#' (character),
#' ssvid (character), engine_power_kw (double), tonnage_gt (double),
#' length_m (double), ais_type (character), event_ais_year (integer),
#' fl_event_id (integer), known_offender (double), known_non_offender (integer)
#' @param gears_interest vector of character elements with names of the gears
#' of interest.
#' @param vars_to_factor vector of character elements with names of columns to
#' convert from character to factor. They have to be columns existing in fl_data
#' and tidy_data.
#' @param vars_remove vector of character elements with names of columns to
#' remove from tidy_data.
#' @return list with 2 elements:
#' holdout_set : data frame with AIS info from offenders before/after
#' the year of offense, potential offenders and known non offenders;
#' training set : data frame with AIS info from offenders during the year
#' of offense, and unlabeled cases
#'
#' @import dplyr
#' @importFrom forcats fct_relevel
#' @importFrom forcats fct_relevel
#'
#' @export
#'


ml_prep_data <- function(tidy_data,
                         gears_interest, vars_to_factor, vars_remove) {

  # filter gears of interest and drop NAs in numeric fields
  tidy_gear <- tidy_data |>
    dplyr::filter(.data$gear %in% gears_interest) |>
    dplyr::filter(!is.na(.data$engine_power_kw) &
                    !is.na(.data$tonnage_gt) &
                    !is.na(.data$length_m))

  # replace remaining NAs in numeric columns with zero
  tidy_na_zero <- tidy_gear |>
    dplyr::select(tidyselect::where(~ is.numeric(.x) && any(is.na(.x)))) |>
    apply(MARGIN = 2, function(x) ifelse(is.na(x), 0, x))
  tidy_gear[, colnames(tidy_na_zero)] <- tidy_na_zero

  # We don't want the same ssvid appearing in different datasets
  # prioritize those for training first, followed by holdout, then unlabeled


  # ssvids reserved for train positives
  train_positive_ssvids <- tidy_gear |>
    dplyr::filter(.data$known_offender == 1,
                  .data$train_validation == 1) |>
    dplyr::distinct(.data$ssvid) |>
    dplyr::pull(.data$ssvid)

  # train_validation positives
  offenders_trainval <- tidy_gear |>
    dplyr::filter(.data$known_offender == 1,
                  .data$train_validation == 1) |>
    dplyr::mutate(
      dataset_group = "train_validation_positive",
      is_holdout = 0
    )

  # holdout positives, excluding any ssvid already used in training
  offenders_holdout <- tidy_gear |>
    dplyr::filter(.data$known_offender == 1,
                  .data$holdout_positive == 1,
                  !.data$ssvid %in% train_positive_ssvids) |>
    dplyr::mutate(
      dataset_group = "holdout_positive",
      is_holdout = 1
    )

  # holdout negatives
  fl_out_non_offenders <- tidy_gear |>
    dplyr::filter(.data$known_non_offender == 1,
                  !.data$ssvid %in% train_positive_ssvids) |>
    dplyr::mutate(
      fl_event_id = as.character(.data$fl_event_id),
      dataset_group = "holdout_negative",
      is_holdout = 1
    )

  # all labeled/prediction ssvids that must be excluded from unlabeled
  reserved_ssvids <- c(
    offenders_trainval |> dplyr::distinct(.data$ssvid) |> dplyr::pull(.data$ssvid),
    offenders_holdout |> dplyr::distinct(.data$ssvid) |> dplyr::pull(.data$ssvid),
    fl_out_non_offenders |> dplyr::distinct(.data$ssvid) |> dplyr::pull(.data$ssvid)
  ) |> unique()

  # unlabeled: no label, and no ssvid used anywhere else
  unlabeled_df <- tidy_gear |>
    dplyr::mutate(sum_fl = .data$known_offender + .data$known_non_offender) |>
    dplyr::filter(.data$sum_fl == 0,
                  !.data$ssvid %in% reserved_ssvids) |>
    dplyr::select(-.data$sum_fl) |>
    dplyr::mutate(
      dataset_group = "unlabeled",
      is_holdout = 0
    )

  # training set: train positives and unlabeled
  training_df <- rbind(offenders_trainval, unlabeled_df) |>
    dplyr::mutate_if(is.logical, as.numeric) |>
    dplyr::mutate(
      source_id = ifelse(.data$known_offender != 1,
                         paste0("no_source_", dplyr::row_number()),
                         .data$source_id),
      fl_event_id = ifelse(.data$known_offender != 1,
                           paste0("no_fl_info_", dplyr::row_number()),
                           .data$fl_event_id)
    ) |>
    dplyr::mutate_at(vars_to_factor, as.factor) |>
    dplyr::mutate(
      known_offender = forcats::fct_relevel(.data$known_offender, c("1", "0"))
    ) |>
    dplyr::mutate(across(tidyselect::where(is.integer), as.numeric)) |>
    dplyr::mutate(indID = paste(.data$ssvid, .data$year, sep = "-"))

  # holdout set: holdout positives and holdout negatives
  holdout_df <- rbind(offenders_holdout, fl_out_non_offenders) |>
    dplyr::select(-tidyselect::any_of(vars_remove)) |>
    dplyr::mutate_if(is.logical, as.numeric) |>
    dplyr::mutate_at(vars_to_factor, as.factor) |>
    dplyr::mutate(
      known_offender = forcats::fct_relevel(.data$known_offender, c("1", "0"))
    ) |>
    dplyr::mutate(across(tidyselect::where(is.integer), as.numeric)) |>
    dplyr::mutate(indID = paste(.data$ssvid, .data$year, sep = "-"))

  return(list(
    holdout_set = holdout_df,
    training_set = training_df
  ))
}
