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
#' @importFrom furrr future_pmap furrr_options
#' @importFrom future cluster
#' @importFrom future multicore
#' @importFrom future multisession
#' @importFrom future plan
#' @importFrom parallel detectCores
#' @importFrom parallel stopCluster
#' @importFrom parallelly makeClusterPSOCK
#' @importFrom parallelly availableCores
#' @importFrom purrr map
#' @importFrom purrr pmap
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
ml_train <- function(fl_rec,
                             rf_spec,
                             cv_splits_all,
                             bag_runs,
                             down_sample_ratio,
                             parallel_plan = "multicore",
                             free_cores = 1,
                             prediction_df = NULL,
                             save_dir = "./models") {

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

  # creating directory
  if (dir.exists(save_dir) == FALSE){
    dir.create(save_dir)
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
    )
    # Make predictions for all CV folds and hyperparameters
    # Run this in parallel, so that each bag is processed on a parallel worker

    furrr::future_pmap(list(train_pred_proba$fl_recipe,
                                            train_pred_proba$common_seed,
                                            train_pred_proba$bag), # previously future_map2, now pmap to map 3 inputs
                                       function(x, y, bag) # added .data$bag as third mapped input
                                       {

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
                                           dplyr::select(-.data$splits)

                                         purrr::pmap(list(cv_predictions$analysis,
                                                          cv_predictions$assessment,
                                                          cv_predictions$id), # was map2 now pmap for 3 inputs
                                                                       function(ind_anal,ind_assess, fold_id) # include id (fold_id)
                                                                       {

                                                                         # Setting seed for seed sampling inside fit
                                                                         set.seed(y)
                                                                         # fit model to analysis data
                                                                         tmp_model <-
                                                                           workflows:::fit.workflow(object = cv_predictions_workflow,
                                                                                                    ind_anal)

                                                                         file_name <- file.path(                       # added filepath to save models with unique names
                                                                           save_dir,
                                                                           paste0("rf_seed", y, "_bag", bag, "_", fold_id, ".rds")
                                                                         )
                                                                         saveRDS(tmp_model, file_name)


                                                                       })
                                       },
                                       .options = furrr::furrr_options(seed = TRUE))

  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }

}
