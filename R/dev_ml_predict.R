#' Predicting over new data using previously trained RF models
#'
#' @param trained_models List of RF models (as returned by dev_ml_train; dev_ml_load)
#' @param new_data New data over which to apply trained models to generate predictions
#' @param free_cores Number of available cores. Add more if you need to do many things at the same time
#' @param parallel_plan Parallelization strategy Options: multisession (if running RStudio), multicore (Linux, Mac and plain R) or psock (if multisession is not working well and you need to try something else)
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
#' @export
#'

dev_ml_predict <- function(trained_models,
                           new_data,
                           free_cores,
                           parallel_plan) {

  # GM: I do not think parallelization is required during prediction
  # Setting up the parallelization
  if (parallel_plan == "multicore") {
    future::plan(future::multicore,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  } else if (parallel_plan == "psock") {
    cl <- parallelly::makeClusterPSOCK(parallelly::availableCores() - free_cores)
    future::plan(future::cluster, workers = cl)
  } else {
    future::plan(future::multisession,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  # Predict with each model
  all_models <- dplyr::bind_rows(trained_models)

  predictions <- purrr::pmap_dfr(
    list(
      all_models$model_id,
      all_models$model_object
    ),
    function(model_id,model_object) {

      # Predict over new data
      tmp_pred <-
        workflows:::predict.workflow(object = model_object,
                                     new_data = new_data,
                                     type = "prob") |>
        dplyr::select(.data$.pred_1) |>
        # Add columns to prediction data
        # (might be a warning about levels in source_id but it's not important,
        # we won't use that column anyway)
        dplyr::bind_cols(new_data[c("indID", "known_offender", "known_non_offender")]) |>
        dplyr::mutate(holdout = 1)

      return(tmp_pred)

    }
  )

  return(predictions)
}
