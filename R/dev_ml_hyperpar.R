#' Get best hyperparameter combination for each common seed after ML training
#'
#' @param data data frame of train cross-validated datasets with
#' several bags, it must have columns:
#' .pred_1 : probability of being an offender;
#' bag: bag ID;
#' known_offender: 0 if not, 1 if yes;
#' .row: row ID
#' common_seed: common seed to generate bags
#' @return data frame of best hyperparameter combinations per common seed
#'
#' @importFrom yardstick roc_auc
#' @import dplyr
#'
#' @export
#'

dev_ml_hyperpar <- function(data) {

  roc_auc_results <- data |>
    dplyr::mutate(counter = as.integer(factor(paste(.data$bag, .data$common_seed)))) |>
    dplyr::group_by(dplyr::across(-c(.data$.pred_1, .data$bag,
                                     .data$known_offender, .data$.row, .data$counter))) |>
    yardstick::roc_auc(truth = .data$known_offender,
                       .data$.pred_1) |>
    dplyr::ungroup() |> # getting auc per hyperparameter combination
    # auc because it's not corrupted by the conditions of our data
    # now we need to get stats across folds per hyperparameter combination
    dplyr::group_by(dplyr::across(-c(.data$id, .data$.estimate))) |>
    # Get mean, min of performance across folds for each hyperparameter
    # Will get NA if fold contains NAs or NaNs
    dplyr::summarize(mean_performance = mean(.data$.estimate),
                     min_performance = min(.data$.estimate)) |>
    dplyr::ungroup()

  # now we need to find the best hyperparameters using the best mean auc per
  # common_seed
  best_hyperparameters <- roc_auc_results |>
    dplyr::arrange(dplyr::desc(.data$mean_performance)) |>
    dplyr::group_by(.data$common_seed) |>
    dplyr::slice(1) |>
    dplyr::select(-.data$.metric, -.data$.estimator, -.data$mean_performance,
                  -.data$min_performance) |>
    dplyr::ungroup()

  return(list(auc_results = roc_auc_results,
              best_hyperparameters = best_hyperparameters))

}
