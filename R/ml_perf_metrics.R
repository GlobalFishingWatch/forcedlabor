
#' Computes recall for assessment sets and specificity for holdout non offenders
#'
#' @description Two performance metrics are computed:
#' recall, for assessment sets (in model versions that did not use them for
#' training), and specificity for holdout non offenders (in the same year of the
#' certification/inspection - if done at the end of the year)
#'
#' @param data tibble with at least a prediction_output column (pred_class),
#' a holdout column (whether if the observation was used in the model or
#' held out), a known_offender column (whether if the vessel was identified as
#' an offender by reports), and a known_non_offender column (whether if the
#' vessel was identified as non offender by inspections).
#' @return tibble with recall and specificity per seed
#'
#' @importFrom purrr pluck
#' @importFrom yardstick recall
#' @importFrom yardstick spec
#' @import dplyr
#'
#' @export
#'

ml_perf_metrics <- function(data) {

  recall_value <- data |>
    dplyr::filter(.data$holdout == 0 ) |>
    yardstick::recall(truth = factor(.data$known_offender,
                                     levels = c(1, 0)),
                      estimate = factor(.data$pred_class,
                                        levels = c(1, 0))) |>
    dplyr::select(.data$.estimate) |>
    purrr::pluck(1)

  specif_value <- data |>
    dplyr::filter(.data$holdout == 1 & .data$known_non_offender == 1) |>
    yardstick::spec(truth = factor(.data$known_offender, levels = c(1, 0)),
                    estimate = factor(.data$pred_class,
                                      levels = c(1, 0))) |>
    dplyr::select(.data$.estimate) |>
    purrr::pluck(1)


    perf_metrics <- data.frame(recall = recall_value, specif = specif_value)


  return(perf_metrics)
}



#' Computes recall for assessment sets
#'
#' @description Computes recall, for assessment sets (in model versions that
#' did not use them for training)
#'
#' @param data tibble with at least a prediction_output
#' column and a known_offender column (whether if the vessel was identified as
#' an offender by reports).
#' @return recall value
#'
#' @importFrom purrr pluck
#' @importFrom yardstick recall
#' @import dplyr
#'
#' @export
#'

ml_recall <- function(data) {

  perf_metrics <- data |>
    yardstick::recall(truth = factor(.data$known_offender,
                                     levels = c(1, 0)),
                      estimate = factor(.data$pred_class,
                                        levels = c(1, 0))) |>
    dplyr::select(.data$.estimate) |>
    purrr::pluck(1)


  return(perf_metrics)
}
