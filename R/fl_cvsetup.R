#' Setting up data structure
#'
#' Sets up the data structure to train the RF model based on a number of folds,
#' bags and common seeds.
#'
#' @param training_data Dataset over which to generate CV folds and tuning grid
#' @param num_folds Number of cross validation folds
#' @param num_bags Number of bags
#' @param num_seeds Number of common seeds
#' @param fl_rec Model recipe
#' @param rf_spec Random forest classifier specifications
#' @param down_sample_ratio See `under_ratio` [themis::step_downsample()].
#' Downsampling ratio to balance the number of positive and unlabelled cases.
#' Defaults to 1 (1:1 ratio)
#' @param group_var A variable in data (single character or name) used for
#' grouping observations with the same value to assign cases to train or test
#' sets within a fold using [rsample::group_vfold_cv()].
#'
#' @returns List object containing cv_folds (analysis/assessment), model
#' workflow and seed/bag identifiers
#'
#' @importFrom dplyr filter mutate row_number select
#' @importFrom purrr map pluck
#' @importFrom rsample group_vfold_cv
#' @importFrom themis step_downsample
#' @importFrom tibble tibble
#' @importFrom tidyr crossing
#' @importFrom workflows add_model add_recipe workflow
#'
#' @export

fl_cvsetup <- function(training_data,
                       fl_rec,
                       rf_spec,
                       num_seeds,
                       num_bags,
                       num_folds,
                       down_sample_ratio,
                       group_var = "source_id") {

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
  # GM: training_data MUST have a source_id_number
  cv_splits_all <- common_seed_tibble |>
    dplyr::mutate(cv_splits = purrr::map(common_seed, function(x) {
      set.seed(x)
      rsample::group_vfold_cv(training_data,
                              group = group_var,
                              v = num_folds)
    }))

  out <- purrr::pmap(list(down_bags$fl_recipe,
                          down_bags$common_seed,
                          down_bags$bag),
                     function(x, y, .bag) {
                       # Ensure all bags look the same
                       set.seed(y)

                       cv_folds <- cv_splits_all |>
                         dplyr::filter(.data$common_seed == y) |>
                         purrr::pluck("cv_splits", 1)

                       return(
                         list(recipe = x,
                              cv_folds = cv_folds,
                              seed = y,
                              bag = .bag)
                       )
                     })

  return(out)
}
