#' Tune random forest
#'
#' Tune random forest hyper-parameters specified during dev_rf_setup
#'
#' @param training_data Dataset over which tune model hyperparameters
#' @param fl_rec Model recipe
#' @param rf_spec Random forest classifier specifications
#' @param num_folds Number of cross validation folds
#' @param num_bags Number of bags
#' @param num_seeds Number of common seeds
#' @param down_sample_ratio See under_ratio in ?themis::step_downsample. To reduce the weight of the unlabeled cases in the model, we randomly
#' downsampled them in the training set with a 1-1 ratio, i.e. the number of
#' positive and unlabeled cases used for training would be equal
#' @param tune_parameters String defining parameter (or parameters) over which to perform tuning. Default to NULL which perform tuning across all hyperparameters in rf_spec
#' @param grid number of grid random values for combinations of hyperparameters per bag, or grid of values to test
#' @param group_var from rsample::group_vfold_cv: A variable in data (single
#' character or name) used for grouping observations with the same value to
#' assign cases to train or test sets within a fold.
#'
#' @returns  Data frame of train cross-validated datasets across hyper-parameter values specified over grid.
#'
#' @importFrom dplyr filter mutate row_number select
#' @importFrom purrr map pluck
#' @importFrom rlang eval_tidy
#' @importFrom rsample group_vfold_cv
#' @importFrom themis step_downsample
#' @importFrom tibble tibble
#' @importFrom tidyr crossing
#' @importFrom tune control_resamples tune tune_grid
#' @importFrom workflows add_model add_recipe workflow
#' @importFrom yardstick metric_set roc_auc
#'
#'
fl_tune <- function(training_data,
                    fl_rec,
                    rf_spec,
                    num_folds,
                    num_bags,
                    num_seeds,
                    down_sample_ratio,
                    tune_parameters = NULL,
                    grid = NULL,
                    group_var) {
  common_seed_tibble <- tibble::tibble(common_seed = seq(1:num_seeds) * 101)

  # Run all common_seeds
  down_bags <- common_seed_tibble |>
    tidyr::crossing(tibble::tibble(bag = seq(num_bags))) |>
    dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) |>
    dplyr::mutate(counter = dplyr::row_number()) |>
    bag_downsample(fl_rec = fl_rec, down_sample_ratio = down_sample_ratio)

  ## Cross Validation
  # Ensure there is no splitting across source_id across analysis and assessment
  # data sets.  Need to make separate splits for each seed.
  cv_splits_all <- common_seed_tibble |>
    dplyr::mutate(cv_splits = purrr::map(common_seed, function(x) {
      set.seed(x)
      rsample::group_vfold_cv(training_data,
                              group = group_var,
                              v = num_folds)
    }))

  if(is.null(grid)) {

    stop("grid must be provided to perform hyper-parameter tuning")

  } else if (!is.null(grid)) {

    params <- if (is.null(tune_parameters)) {
      c("trees", "mtry", "min_n","regularization.factor")
    } else {
      tune_parameters
    }

    trees_val <- if ("trees" %in% params) tune::tune() else rlang::eval_tidy(rf_spec$args$trees)
    mtry_val  <- if ("mtry"  %in% params) tune::tune() else rlang::eval_tidy(rf_spec$args$mtry)
    min_n_val <- if ("min_n" %in% params) tune::tune() else rlang::eval_tidy(rf_spec$args$min_n)
    reg_val   <- if ("regularization.factor" %in% params) tune::tune() else rlang::eval_tidy(rf_spec$eng_args$regularization.factor)

    message("Performing parameter tuning for ", paste(params, collapse = ", "), " over specified grid")

    rf_spec <- rf_spec |>
      recipes::update(
        trees = !!trees_val,
        mtry  = !!mtry_val,
        min_n = !!min_n_val,
        regularization.factor = !!reg_val
      )

  }

  out <- furrr::future_pmap(list(down_bags$fl_recipe,
                                 down_bags$common_seed,
                                 down_bags$bag),
                            function(x, y, .bag) {
                              set.seed(y)
                              cv_splits <- cv_splits_all |>
                                dplyr::filter(.data$common_seed == y) |>
                                purrr::pluck('cv_splits', 1)

                # specifying the workflow, recipe for data and tuning
                              cv_predictions <- workflows::workflow() |>
                                workflows::add_model(rf_spec) |>
                                workflows::add_recipe(x) |>
                      # Automatically creates hyperparameter grid
                      # using a space-filling design (via a Latin hypercube)
                                tune::tune_grid(resamples = cv_splits,
                                                grid = grid,
                                                metrics = yardstick::metric_set(yardstick::roc_auc),
                                                control = tune::control_resamples(save_pred = TRUE)) |>
                                dplyr::select(id, .data$.predictions) |>
                                tidyr::unnest(.data$.predictions) |>
                                dplyr::select(-.data$.pred_0, -.data$.config) |>
                                dplyr::mutate(bag = .bag,
                                              common_seed = y)

                              return(
                                cv_predictions
                              )
                            },.options = furrr::furrr_options(seed = TRUE, packages = c("themis")))
  return(do.call(rbind,out))
}
