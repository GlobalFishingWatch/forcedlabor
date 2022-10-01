
#' Computes recall for assessment sets and specificity for holdout non offenders
#'
#' @description For each common seed, two performance metrics are computed:
#' recall, for assessment sets (in model versions that did not use them for
#' training), and specificity for holdout non offenders (in the same year of the
#' certification/inspection - if done at the end of the year)
#'
#' @param data tibble with at least a common_seed column, a prediction_output
#' column, a holdout column (whether if the observation was used in the model or
#' held out), a known_offender column (whether if the vessel was identified as
#' an offender by reports), and a known_non_offender column (whether if the
#' vessel was identified as non offender by inspections).
#' @param common_seed_tibble tibble with one column containing all the common
#' seeds
#' @param specif If TRUE (default), specificity is computed
#' @return tibble with recall and specificity per seed
#'
#' @importFrom purrr map_dbl
#' @importFrom purrr pluck
#' @importFrom yardstick recall
#' @importFrom yardstick spec
#' @import dplyr
#'
#' @export
#'

ml_perf_metrics <- function(data, common_seed_tibble, specif = TRUE) {

  recall_seed <- common_seed_tibble %>%
    dplyr::mutate(recall_perf = purrr::map_dbl(.data$common_seed, function(x) {
      data %>%
        dplyr::filter(.data$holdout == 0 & .data$common_seed == x) %>%
        yardstick::recall(truth = factor(.data$known_offender,
                                         levels = c(1, 0)),
                          estimate = factor(.data$pred_class,
                                            levels = c(1, 0))) %>%
        dplyr::select(.data$.estimate) %>%
        purrr::pluck(1)
    }))

  if (specif == TRUE){

    specif_seed <- common_seed_tibble %>%
      dplyr::mutate(spec_perf = purrr::map_dbl(.data$common_seed, function(x) {
        data %>%
          dplyr::filter(.data$holdout == 1 & .data$known_non_offender == 1 &
                          .data$event_ais_year == 1 & .data$common_seed == x) %>%
          yardstick::spec(truth = factor(.data$known_offender, levels = c(1, 0)),
                          estimate = factor(.data$pred_class,
                                            levels = c(1, 0))) %>%
          dplyr::select(.data$.estimate) %>%
          purrr::pluck(1)
      }))

    perf_metrics <- recall_seed %>%
      dplyr::full_join(specif_seed, by = "common_seed")

  }else{
    perf_metrics <- recall_seed
  }

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
#' @importFrom purrr map_dbl
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



ml_perf_metrics_composite <- function(data) {

   recall_stat <-  data %>%
        dplyr::filter(.data$holdout == 0) %>%
        yardstick::recall(truth = factor(.data$known_offender,
                                         levels = c(1, 0)),
                          estimate = factor(.data$class_mode,
                                            levels = c(1, 0))) %>%
        dplyr::select(.data$.estimate) %>%
        purrr::pluck(1)

  specif_stat <- data %>%
        dplyr::filter(.data$holdout == 1 & .data$known_non_offender == 1 &
                        .data$event_ais_year == 1) %>%
        yardstick::spec(truth = factor(.data$known_offender, levels = c(1, 0)),
                        estimate = factor(.data$class_mode,
                                          levels = c(1, 0))) %>%
        dplyr::select(.data$.estimate) %>%
        purrr::pluck(1)


  perf_metrics <- cbind(recall = recall_stat, specif = specif_stat)

  return(perf_metrics)
}
