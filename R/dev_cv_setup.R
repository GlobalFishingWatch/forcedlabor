#' Setting up the data structure to train the RF model based on a number of folds, bags and common seeds.
#' If tune is set to TRUE, is provided  return..., otherwise return analysis/assessment splits and model workflows
#' across CV folds and bags.
#'
#' @param training_data Dataset over which to generate CV folds and tuning grid
#' @param num_folds Number of cross validation folds
#' @param num_bags Number of bags
#' @param num_seeds Number of common seeds
#' @param fl_rec Model recipe
#' @param rf_spec Random forest classifier specifications
#' @param down_sample_ratio See under_ratio in ?themis::step_downsample. To reduce the weight of the unlabeled cases in the model, we randomly
#' downsampled them in the training set with a 1-1 ratio, i.e. the number of
#' positive and unlabeled cases used for training would be equal
#' @param tune Boolean defining wether to perform hyperparameter tuning in rf_spec
#' @param tune_parameters String defining parameter (or parameters) over which to perform tuning. Default to NULL which perform tuning across all hyperparameters in rf_spec
#' @param grid Grid defining hyperparameters values in rf_spec over which to perform tuning. Must contain the same parameters specified in tune_parameters
#'
#' @returns List object containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers
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
#' @export

dev_cv_setup <- function(training_data,
                          num_folds,
                          num_bags,
                          num_seeds,
                          fl_rec,
                          rf_spec,
                          down_sample_ratio,
                          tune = FALSE,
                          tune_parameters = NULL,
                          grid = NULL)
{

  common_seed_tibble <- tibble::tibble(common_seed =
                                         seq(1:num_seeds) * 101)

  # Run all common_seeds
  # GM: probably merge dev_bag_downsample with this pipe
  bag_runs <- common_seed_tibble |>
    tidyr::crossing(tibble::tibble(bag = seq(num_bags))) |>
    dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) |>
    dplyr::mutate(counter = dplyr::row_number())

  down_bags<-dev_bag_downsample(bag_runs = bag_runs,
                                fl_rec = fl_rec,
                                down_sample_ratio = down_sample_ratio)

  ## Cross Validation
  # Ensure there is no splitting across source_id across analysis and assessment
  # data sets.  Need to make separate splits for each seed.
  # GM: training_data MUST have a source_id_number
  cv_splits_all <- common_seed_tibble |>
    dplyr::mutate(cv_splits = purrr::map(common_seed, function(x) {
      set.seed(x)
      rsample::group_vfold_cv(training_data,
                              group = source_id_number,
                              v = num_folds)
    }))

  if(tune && is.null(grid)){

    stop("grid must be provided when tune = TRUE")

  } else if (tune && !is.null(grid)) {

    params <- if (tune && is.null(tune_parameters)) {
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
      update(
        trees = !!trees_val,
        mtry  = !!mtry_val,
        min_n = !!min_n_val,
        regularization.factor = !!reg_val
      )

  } else if (!tune) {

    message("Generating analysis/asessment datasets across folds")

  }

  out<-purrr::pmap(list(down_bags$fl_recipe,
                        down_bags$common_seed,
                        down_bags$bag), # previously future_map2, now pmap to map 3 inputs
                   function(x, y, .bag) # added .data$bag as third mapped input
                   {

                     # Ensure all bags look the same
                     set.seed(y)

                     if (tune && !is.null(grid)) {
                       # GM: I am not sure what the tune::tune_grid is doing

                       cv_splits <- cv_splits_all |>
                         dplyr::filter(.data$common_seed == y) |>
                         purrr::pluck('cv_splits')  |>
                         purrr::pluck(1) # unlist first (unique) element
                       # specifying the workflow with the model, recipe for data and how the
                       # tuning goes

                       cv_predictions <- workflows::workflow() |>
                         workflows::add_model(rf_spec) |>
                         workflows::add_recipe(x) |>
                         tune::tune_grid(resamples = cv_splits,
                                         # Automatically creates hyperparameter grid
                                         # using a space-filling design (via a Latin hypercube)
                                         grid = grid,
                                         # Need to specify a metric to calculate, even though we
                                         # won't use it for anything
                                         # Doing ROC means that the predictions this outputs will be
                                         # the raw numeric, rather than class
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

                       } else if (!tune) {

                       # specifying the workflow with the model, recipe for data and how the
                       # tuning goes
                       cv_predictions_workflow <-
                         workflows::workflow() |>
                         workflows::add_model(rf_spec) |> #GM: is this necessary here, or could be included later in dev_ml_train once hyperparameters defined?
                         workflows::add_recipe(x)

                       # get the folds related to that common seed, train and predict
                       cv_predictions <-
                         cv_splits_all |>
                         dplyr::filter(.data$common_seed == y) |>
                         purrr::pluck('cv_splits',1) |> # unlist first (unique) element
                         dplyr::mutate(# Create analysis dataset based on CV folds
                           analysis = purrr::map(.data$splits, ~rsample::analysis(.x)),
                           # Create assessment dataset based on CV folds
                           assessment = purrr::map(.data$splits, ~rsample::assessment(.x))) |>
                         dplyr::select(-.data$splits)

                       return(
                         list(
                           workflow   = cv_predictions_workflow,
                           cv_folds = cv_predictions,
                           seed = y,
                           bag = .bag
                         )
                       )
                     }
                   })

  return(out)
}
