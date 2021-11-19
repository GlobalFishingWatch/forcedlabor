#' Training and predicting with PU learning
#' Ideally do all ml_training, ml_hyperpar, ml_training_Frank_2, dedpul, assessments
#'
#' @param training_df data frame. training set.
#' @param fl_rec recipe
#' @param rf_spec model specifications
#' @param cv_splits_all tibble containing tibbles of cross-validation splits. 1 cv split per common_seed
#' @param bag_runs data frame with the ID of each bag for each common seed and
#' the actual seed using to downsample in each bag
#' @param down_sample_ratio down sampling ratio for the smaller class (offenders);
#' to add to the recipe in each bag
#' @param num_grid number of grid random values for combinations of
#' hyperparameters per bag
#' @param parallel_plan type of parallelization to run (multicore, multisession
#' or psock - this last one may need calling libraries inside)
#' @param free_cores number of free cores to leave out of parallelization
#'
ml_train_predict <- function(training_df, fl_rec, rf_spec, cv_splits_all,
                             bag_runs, down_sample_ratio = 1, num_grid = 5,
                             parallel_plan = "multicore", free_cores = 1){


  train_pred_proba <- ml_training(training_df, fl_rec, rf_spec, cv_splits_all,
                                  bag_runs, down_sample_ratio, num_grid,
                                  parallel_plan, free_cores)


  return(train_pred_proba)

}


#' Trains machine learning (RF) models:
#'
#' @description For each bag seed, it generates a set of model hyperparameter configurations,
# fits a downsampled set of data to each model for each bag and each fold,
# and predicts over its corresponding assessment set (for each fold)
#'
#' @param training_df data frame. training set.
#' @param fl_rec recipe
#' @param rf_spec model specifications
#' @param cv_splits_all tibble containing tibbles of cross-validation splits. 1 cv split per common_seed
#' @param bag_runs data frame with the ID of each bag for each common seed and
#' the actual seed using to downsample in each bag
#' @param down_sample_ratio down sampling ratio for the smaller class (offenders);
#' to add to the recipe in each bag
#' @param num_grid number of grid random values for combinations of
#' hyperparameters per bag
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
#' @importFrom purrr map
#' @importFrom purrr pluck
#' @importFrom themis step_downsample
#' @importFrom tidyr unnest
#' @importFrom tune tune_grid
#' @importFrom workflows workflow
#' @importFrom workflows add_model
#' @importFrom workflows add_recipe
#'
#' @export
#'

ml_training <- function(training_df, fl_rec, rf_spec, cv_splits_all, #common_seed_tibble,
                        # num_folds,
                        bag_runs, down_sample_ratio, num_grid = 5,
                        parallel_plan = "multicore", free_cores = 1){

  # Setting up the parallelization
  if (parallel_plan == "multicore"){
    future::plan(multicore, workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  }else if (parallel_plan == "psock"){
    cl <- parallelly::makeClusterPSOCK(availableCores() - free_cores)
    future::plan(cluster, workers = cl)
  }else{
    utils::globalVariables("multisession")
    future::plan(future:::multisession, workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  # we train and predict probabilities of being an offender during cross-validation

  train_pred_proba <- bag_runs %>%
    dplyr::mutate(
      # get a recipe with downsampling for each bag and corresponding seed
      fl_recipe = purrr::map(recipe_seed, function(x){
        fl_rec_down <- fl_rec %>%
          themis::step_downsample(known_offender,
                                  under_ratio = down_sample_ratio, seed = x,
                                  skip = TRUE) #%>%
      })
    ) %>%
    # Make predictions for all CV folds and hyperparameters
    # Run this in parallel, so that each bag is processed on a parallel worker
    dplyr::mutate(predictions = furrr::future_map2(fl_recipe, common_seed, function(x,y){
      # Ensure all bags look the same across hyperparameter tuning grid
      #
      set.seed(y)
      # library(recipeselectors)
      cv_splits <- cv_splits_all %>%
        dplyr::filter(common_seed==y) %>%
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
                  metrics = metric_set(roc_auc),
                  control = control_resamples(save_pred = TRUE)) %>%
        dplyr::select(id,.predictions) %>%
        tidyr::unnest(.predictions)%>%
        dplyr::select(-.pred_0,-.config)
      return(cv_predictions)
    },.options=furrr::furrr_options(seed=TRUE))) %>%
    # Remove unnecessary columns
    dplyr::select(-recipe_seed,-fl_recipe)%>%
    tidyr::unnest(predictions)

  if (parallel_plan == "psock"){
    parallel::stopCluster(cl)
  }
  # future:::ClusterRegistry("stop") # use this in case there's an error in
  # the code and it stops abruptly without collecting garbage


  return(train_pred_proba)
}
