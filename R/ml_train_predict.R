#' Trains machine learning (RF) models:
#'
#' @description For each bag seed, it generates a set of model hyperparameter
#' configurations,
# fits a downsampled set of data to each model for each bag and each fold,
# and predicts over its corresponding assessment set (for each fold)
#'
#' @param training_df data frame. training set.
#' @param fl_rec recipe
#' @param rf_spec model specifications
#' @param cv_splits_all tibble containing tibbles of cross-validation splits.
#' 1 cv split per common_seed
#' @param bag_runs data frame with the ID of each bag for each common seed and
#' the actual seed using to downsample in each bag
#' @param down_sample_ratio down sampling ratio for the smaller class
#' (offenders); to add to the recipe in each bag
#' @param num_grid number of grid random values for combinations of
#' hyperparameters per bag, or grid of values to test
#' @param parallel_plan type of parallelization to run (multicore, multisession
#' or psock - this last one may need calling libraries inside)
#' @param free_cores number of free cores to leave out of parallelization
#' @return data frame with confidence scores of being an offender
#'
#' @importFrom dplyr mutate
#' @importFrom dplyr filter
#' @importFrom dplyr select
#' @importFrom furrr furrr_options
#' @importFrom future plan
#' @importFrom parallel detectCores
#' @importFrom parallel stopCluster
#' @importFrom parallelly makeClusterPSOCK
#' @importFrom parallelly availableCores
#' @importFrom purrr map
#' @importFrom purrr pluck
#' @importFrom themis step_downsample
#' @importFrom tidyr unnest
#' @importFrom tune tune_grid
#' @importFrom tune control_resamples
#' @importFrom workflows workflow
#' @importFrom workflows add_model
#' @importFrom workflows add_recipe
#' @importFrom yardstick metric_set
#' @importFrom yardstick roc_auc
#' @rawNamespace import(recipes, except = step_downsample)
#' @import tidyselect
#'
#' @export
#'

ml_training <- function(training_df, fl_rec, rf_spec, cv_splits_all,
                        bag_runs, down_sample_ratio, num_grid = 5,
                        parallel_plan = "multicore", free_cores = 1) {

  # Setting up the parallelization
  if (parallel_plan == "multicore") {
    future::plan(future::multicore,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  }else if (parallel_plan == "psock") {
    cl <- parallelly::makeClusterPSOCK(parallelly::availableCores() - free_cores)
    future::plan(future::cluster, workers = cl)
  }else {
    future::plan(future::multisession,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  # we train and predict probabilities of being an offender during
  # cross-validation

  train_pred_proba <- bag_runs %>%
    dplyr::mutate(
      # get a recipe with downsampling for each bag and corresponding seed
      fl_recipe = purrr::map(.data$recipe_seed, function(x) {
        fl_rec_down <- fl_rec %>%
          themis::step_downsample(known_offender,
                                  under_ratio = down_sample_ratio, seed = x,
                                  skip = TRUE) #%>%
      })
    ) %>%
    # Make predictions for all CV folds and hyperparameters
    # Run this in parallel, so that each bag is processed on a parallel worker
    dplyr::mutate(predictions = furrr::future_map2(.data$fl_recipe,
                                                   .data$common_seed,
                                                   function(x, y) {
                                                     # Ensure all bags look the same across hyperparameter tuning grid
                                                     set.seed(y)
                                                     cv_splits <- cv_splits_all %>%
                                                       dplyr::filter(.data$common_seed == y) %>%
                                                       .$cv_splits %>%
                                                       purrr::pluck(1) # unlist first (unique) element

                                                     # specifying the workflow with the model, recipe for data and how the
                                                     # tuning goes
                                                     cv_predictions <- workflows::workflow() %>%
                                                       workflows::add_model(rf_spec) %>%
                                                       workflows::add_recipe(x) %>%
                                                       tune::tune_grid(resamples = cv_splits,
                                                                       # Automatically creates hyperparameter grid
                                                                       # using a space-filling design (via a Latin hypercube)
                                                                       grid = num_grid,
                                                                       # Need to specify a metric to calculate, even though we
                                                                       # won't use it for anything
                                                                       # Doing ROC means that the predictions this outputs will be
                                                                       # the raw numeric, rather than class
                                                                       metrics = yardstick::metric_set(yardstick::roc_auc),
                                                                       control = tune::control_resamples(save_pred = TRUE)) %>%
                                                       dplyr::select(id, .data$.predictions) %>%
                                                       tidyr::unnest(.data$.predictions) %>%
                                                       dplyr::select(-.data$.pred_0, -.data$.config)
                                                     return(cv_predictions)
                                                   }, .options = furrr::furrr_options(seed = TRUE))) %>%
    # Remove unnecessary columns
    dplyr::select(-.data$recipe_seed, -.data$fl_recipe) %>%
    tidyr::unnest(.data$predictions)

  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }

  return(train_pred_proba)
}


#' Get best hyperparameter combination for each common seed after ML training
#'
#' @param train_pred_proba data frame of train cross-validated datasets with
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

ml_hyperpar <- function(train_pred_proba) {

  roc_auc_results <- train_pred_proba %>%
    dplyr::group_by(dplyr::across(-c(.data$.pred_1, .data$bag,
                                     .data$known_offender, .data$.row, .data$counter))) %>%
    yardstick::roc_auc(truth = .data$known_offender,
                       .data$.pred_1) %>%
    dplyr::ungroup() %>% # getting auc per hyperparameter combination
    # auc because it's not corrupted by the conditions of our data
    # now we need to get stats across folds per hyperparameter combination
    dplyr::group_by(dplyr::across(-c(.data$id, .data$.estimate))) %>%
    # Get mean, min of performance across folds for each hyperparameter
    # Will get NA if fold contains NAs or NaNs
    dplyr::summarize(mean_performance = mean(.data$.estimate),
                     min_performance = min(.data$.estimate)) %>%
    dplyr::ungroup()

  # now we need to find the best hyperparameters using the best mean auc per
  # common_seed
  best_hyperparameters <- roc_auc_results %>%
    dplyr::arrange(dplyr::desc(.data$mean_performance)) %>%
    dplyr::group_by(.data$common_seed) %>%
    dplyr::slice(1) %>%
    dplyr::select(-.data$.metric, -.data$.estimator, -.data$mean_performance,
                  -.data$min_performance) %>%
    dplyr::ungroup()

  return(list(auc_results = roc_auc_results,
              best_hyperparameters = best_hyperparameters))

}


#' Trains machine learning (RF) models and predicts
#'
#' @description For each bag seed, fit one random forest to each bag in each
#' fold, and predict over the assessment set and the holdout set.
#'
#' @param training_df training data frame
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


# Trains machine learning (RF with fixed hyperparameters) models and predicts


# Trains machine learning (RF with fixed hyperparameters) models and predicts

ml_train_predict <- function(training_df, fl_rec, rf_spec, cv_splits_all,
                             bag_runs, down_sample_ratio,
                             parallel_plan = "multicore", free_cores = 1,
                             prediction_df = NULL) {

  # Setting up the parallelization
  if (parallel_plan == "multicore") {
    future::plan(future::multicore,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  }else if (parallel_plan == "psock") {
    cl <- parallelly::makeClusterPSOCK(parallelly::availableCores() - free_cores)
    future::plan(future::cluster, workers = cl)
  }else {
    future::plan(future::multisession,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  # here we train and predict probabilities of being an offender during
  # cross-validation

  if (is.null(prediction_df) == FALSE){

    train_pred_proba <-
      bag_runs |>
      dplyr::mutate(
        # get a recipe with downsampling for each bag and corresponding seed
        fl_recipe = purrr::map(.data$recipe_seed, function(x) {
          fl_rec_down <- fl_rec |>
            themis::step_downsample(known_offender,
                                    under_ratio = down_sample_ratio, seed = x,
                                    skip = TRUE) #%>%
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
                                               analysis = purrr::map(.data$splits,~rsample::analysis(.x)),
                                               # Create assessment dataset based on CV folds
                                               assessment = purrr:::map(.data$splits,~rsample::assessment(.x))) |>
                                             dplyr::select(-.data$splits) |>
                                             dplyr::mutate(predictions =
                                                             purrr::map2(analysis,assessment,function(ind_anal,ind_assess){
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

                                                               return(tmp_pred)

                                                             })) |>
                                             dplyr::select(.data$id, .data$predictions) |>
                                             tidyr::unnest(.data$predictions)

                                           return(cv_predictions)
                                         }, .options = furrr::furrr_options(seed = TRUE))) |>

      # Remove unnecessary columns
      dplyr::select(-.data$recipe_seed, -.data$fl_recipe) |>
      tidyr::unnest(.data$predictions)


  }else{

    train_pred_proba <-
      bag_runs |>
      dplyr::mutate(
        # get a recipe with downsampling for each bag and corresponding seed
        fl_recipe = purrr::map(.data$recipe_seed, function(x) {
          fl_rec_down <- fl_rec |>
            themis::step_downsample(known_offender,
                                    under_ratio = down_sample_ratio, seed = x,
                                    skip = TRUE) #%>%
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
                                               analysis = purrr::map(.data$splits,~rsample::analysis(.x)),
                                               # Create assessment dataset based on CV folds
                                               assessment = purrr:::map(.data$splits,~rsample::assessment(.x))) |>
                                             dplyr::select(-.data$splits) |>
                                             dplyr::mutate(predictions =
                                                             purrr::map2(analysis,assessment,function(ind_anal,ind_assess){
                                                               # Setting seed for seed sampling inside fit
                                                               set.seed(y)
                                                               # fit model to analysis data
                                                               tmp_model <-
                                                                 workflows:::fit.workflow(object = cv_predictions_workflow,
                                                                                          ind_anal)
                                                               # Predict over assessment data using fit
                                                               tmp_pred <-
                                                                 workflows:::predict.workflow(object = tmp_model,
                                                                                              new_data = ind_assess,
                                                                                              type = "prob") |>
                                                                 dplyr::select(.data$.pred_1) |>
                                                                 # Add columns to assessment data
                                                                 dplyr::bind_cols(ind_assess[c("indID", "known_offender", "known_non_offender")]) |>
                                                                 dplyr::mutate(holdout = 0)

                                                               return(tmp_pred)

                                                             })) |>
                                             dplyr::select(.data$id, .data$predictions) |>
                                             tidyr::unnest(.data$predictions)

                                           return(cv_predictions)
                                         }, .options = furrr::furrr_options(seed = TRUE))) |>

      # Remove unnecessary columns
      dplyr::select(-.data$recipe_seed, -.data$fl_recipe) |>
      tidyr::unnest(.data$predictions)

  }


  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }



  return(models_pred = train_pred_proba)

}
