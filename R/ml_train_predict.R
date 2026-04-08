#' Trains machine learning (RF) models and predicts
#'
#' @description For each bag seed, fit one random forest to each bag in each
#' fold, and predict over the assessment set and the holdout set.
#'
#' @param fl_rec recipe
#' @param rf_spec model specifications
#' @param cv_splits_all tibble containing tibbles of cross-validation splits
#' (1 split per common seed)
#' @param bag_runs bags
#' @param down_sample_ratio down sampling ratio for the predicted class to add
#' to the recipe in each bag
#' @param parallel_plan type of parallelization to run (multicore, multisession,
#' or psock - this last one may need calling libraries inside)
#' @param free_cores number of free cores to leave out of parallelization
#' @param prediction_df hold-out data frame with possible offenders and non
#' offenders to predict on. If NULL (default), then only predict on the training set
#' @return an object with predicted values and fitted models
#'
#' @importFrom furrr future_map2
#' @importFrom future cluster
#' @importFrom future multicore
#' @importFrom future multisession
#' @importFrom future plan
#' @importFrom parallel detectCores
#' @importFrom parallel stopCluster
#' @importFrom parallelly makeClusterPSOCK
#' @importFrom parallelly availableCores
#' @importFrom purrr map
#' @importFrom purrr map2
#' @importFrom purrr pluck
#' @importFrom rsample analysis
#' @importFrom rsample assessment
#' @importFrom themis step_downsample
#' @importFrom tidyr unnest
#' @importFrom workflows add_model
#' @importFrom workflows add_recipe
#' @importFrom workflows workflow
#' @import dplyr
#'
#' @export
#'
ml_train_predict <- function(fl_rec,
                             rf_spec,
                             cv_splits_all,
                             bag_runs,
                             down_sample_ratio,
                             parallel_plan = "multicore",
                             free_cores = 1,
                             prediction_df = NULL) {

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

  # here we train and predict probabilities of being an offender during
  # cross-validation
  train_pred_proba <-
    bag_runs |>
    dplyr::mutate(
      # get a recipe with downsampling for each bag and corresponding seed
      fl_recipe = purrr::map(.data$recipe_seed, function(x) {
        fl_rec_down <- fl_rec |>
          themis::step_downsample(known_offender,
                                  under_ratio = down_sample_ratio,
                                  seed = x,
                                  skip = TRUE)
      })
    ) |>
    # Make predictions for all CV folds and hyperparameters
    # Run this in parallel, so that each bag is processed on a parallel worker
    dplyr::mutate(predictions =
                    furrr::future_map2(.data$fl_recipe,
                                       .data$common_seed,
                                       function(x, y) {
                                         # Ensure all bags look the same
                                         set.seed(y)
                                         # specifying the workflow with the model, recipe for data and how the
                                         # tuning goes
                                         cv_predictions_workflow <-
                                           workflows::workflow() |>
                                           workflows::add_model(rf_spec) |>
                                           workflows::add_recipe(x)

                                         # get the folds related to that common seed, train and predict
                                         cv_predictions <-
                                           cv_splits_all |>
                                           dplyr::filter(.data$common_seed == y) |>
                                           purrr::pluck('cv_splits') |>
                                           # .$cv_splits |>
                                           purrr::pluck(1) |>  # unlist first (unique) element
                                           dplyr::mutate(# Create analysis dataset based on CV folds
                                             analysis = purrr::map(.data$splits, ~rsample::analysis(.x)),
                                             # Create assessment dataset based on CV folds
                                             assessment = purrr::map(.data$splits, ~rsample::assessment(.x))) |>
                                           dplyr::select(-.data$splits) |>
                                           dplyr::mutate(predictions =
                                                           purrr::map2(analysis,
                                                                       assessment,
                                                                       function(ind_anal,ind_assess) {
                                                                         # Setting seed for seed sampling inside fit
                                                                         set.seed(y)
                                                                         # fit model to analysis data
                                                                         tmp_model <-
                                                                           workflows:::fit.workflow(object = cv_predictions_workflow,
                                                                                                    ind_anal)
                                                                         # Predict over assessment data using fit
                                                                         tmp_pred_assess <-
                                                                           workflows:::predict.workflow(object = tmp_model,
                                                                                                        new_data = ind_assess,
                                                                                                        type = "prob") |>
                                                                           dplyr::select(.data$.pred_1) |>
                                                                           # Add columns to assessment data
                                                                           dplyr::bind_cols(ind_assess[c("indID", "known_offender", "known_non_offender")]) |>
                                                                           dplyr::mutate(holdout = 0)

                                                                         if (is.null(prediction_df) == FALSE) { #bringing the logical clause here
                                                                           # # Predict over data not used for training
                                                                           tmp_pred <-
                                                                             workflows:::predict.workflow(object = tmp_model,
                                                                                                          new_data = prediction_df,
                                                                                                          type = "prob") |>
                                                                             dplyr::select(.data$.pred_1) |>
                                                                             # Add columns to prediction data
                                                                             # (might be a warning about levels in source_id but it's not important,
                                                                             # we won't use that column anyway)
                                                                             dplyr::bind_cols(prediction_df[c("indID", "known_offender", "known_non_offender")]) |>
                                                                             dplyr::mutate(holdout = 1) |>
                                                                             dplyr::bind_rows(tmp_pred_assess)
                                                                         } else {

                                                                           # Predict over assessment data using fit
                                                                           tmp_pred <-
                                                                             workflows:::predict.workflow(object = tmp_model,
                                                                                                          new_data = ind_assess,
                                                                                                          type = "prob") |>
                                                                             dplyr::select(.data$.pred_1) |>
                                                                             # Add columns to assessment data
                                                                             dplyr::bind_cols(ind_assess[c("indID", "known_offender", "known_non_offender")]) |>
                                                                             dplyr::mutate(holdout = 0)
                                                                           # no bind_rows?
                                                                         }

                                                                         return(tmp_pred)

                                                                       })) |>
                                           dplyr::select(.data$id, .data$predictions) |>
                                           tidyr::unnest(.data$predictions)

                                         return(cv_predictions)
                                       },
                                       .options = furrr::furrr_options(seed = TRUE))) |>

    # Remove unnecessary columns
    dplyr::select(-.data$recipe_seed, -.data$fl_recipe) |>
    tidyr::unnest(.data$predictions)



  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }



  return(models_pred = train_pred_proba)

}
