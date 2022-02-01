
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

ml_perf_metrics <- function(data, common_seed_tibble) {

  recall_seed <- common_seed_tibble %>%
    dplyr::mutate(recall_perf = purrr::map_dbl(common_seed, function(x) {
      data %>%
        dplyr::filter(holdout == 0 & common_seed == x) %>%
        yardstick::recall(truth = factor(known_offender, levels = c(1, 0)),
                          estimate = factor(pred_class, levels = c(1, 0))) %>%
        dplyr::select(.estimate) %>%
        purrr::pluck(1)
    }))

  specif_seed <- common_seed_tibble %>%
    dplyr::mutate(spec_perf = purrr::map_dbl(common_seed, function(x) {
      data %>%
        dplyr::filter(holdout == 1 & known_non_offender == 1 &
                        event_ais_year == 1 & common_seed == x) %>%
        yardstick::spec(truth = factor(known_offender, levels = c(1, 0)),
                        estimate = factor(pred_class, levels = c(1, 0))) %>%
        dplyr::select(.estimate) %>%
        purrr::pluck(1)
    }))

  perf_metrics <- recall_seed %>%
    dplyr::full_join(specif_seed, by = "common_seed")

  return(perf_metrics)
}
