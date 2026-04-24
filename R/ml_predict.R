#' Predicting over new data using previously trained RF models
#'
#' @param trained_models List of RF models (as returned by dev_ml_train; dev_ml_load)
#' @param new_data New data over which to apply trained models to generate predictions
#'
#' @returns Dataframe with predicted values for each RF model
#'
#' @importFrom dplyr bind_cols bind_rows mutate select
#' @importFrom future cluster multicore multisession plan
#' @importFrom parallel detectCores
#' @importFrom parallelly availableCores makeClusterPSOCK
#' @importFrom purrr pmap_dfr
#' @import workflows
#'
#'

ml_predict <- function(trained_models,
                       new_data) {

  # Predict with each model
  all_models <- dplyr::bind_rows(trained_models)

  predictions <- purrr::pmap_dfr(
    list(
      all_models$seed,
      all_models$bag,
      all_models$fold_id,
      all_models$model_object
    ),
    function(seed, .bag, fold_id, model_object) {

      # Predict over new data
      tmp_pred <-
        workflows:::predict.workflow(object = model_object,
                                     new_data = new_data,
                                     type = "prob") |>
        dplyr::select(.data$.pred_1) |>
        dplyr::bind_cols(new_data[c("indID", "known_offender", "known_non_offender")]) |>
        dplyr::mutate(holdout = 1,
                      common_seed = seed,
                      bag = .bag,
                      id = fold_id)
      return(tmp_pred)

    }
  )

  return(predictions)
}
