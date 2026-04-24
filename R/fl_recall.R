#' Computes recall for assessment sets
#'
#' @description Computes recall, for assessment sets (in model versions that
#' did not use them for training)
#'
#' @param data tibble with at least a prediction output column (`pred_class`)
#' and a `known_offender` column (whether the vessel was identified as
#' an offender by reports).
#' @return recall value
#'
#' @importFrom purrr pluck
#' @importFrom yardstick recall
#' @import dplyr
#'
#' @export
#'

fl_recall <- function(data) {

  perf_metrics <- data |>
    yardstick::recall(truth = factor(.data$known_offender,
                                     levels = c(1, 0)),
                      estimate = factor(.data$pred_class,
                                        levels = c(1, 0))) |>
    dplyr::select(.data$.estimate) |>
    purrr::pluck(1)


  return(perf_metrics)
}
