#' Setting up data structure
#'
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
#' downsampled them in the training set with a 1-1 ratio, i.e. the number of positive and unlabeled cases used for training would be equal
#' @param group_var from rsample::group_vfold_cv: A variable in data (single character or name) used for grouping observations with the same value
#' to either the analysis or assessment set within a fold.
#'
#' @returns List object containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers
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

dev_cv_setup2 <- function(training_data,
                         num_folds,
                         num_bags,
                         num_seeds,
                         fl_rec,
                         rf_spec,
                         down_sample_ratio,
                         group_var = "source_id_number") {

  common_seed_tibble <- tibble::tibble(common_seed =
                                         seq(1:num_seeds) * 101)

  # Run all common_seeds
  down_bags <- common_seed_tibble |>
    tidyr::crossing(tibble::tibble(bag = seq(num_bags))) |>
    dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) |>
    dplyr::mutate(counter = dplyr::row_number()) |>
    dev_bag_downsample(fl_rec = fl_rec,
                       down_sample_ratio = down_sample_ratio)

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

  out <- purrr::pmap(list(down_bags$fl_recipe,
                         down_bags$common_seed,
                         down_bags$bag),
                     function(x, y, .bag) {
                       # Ensure all bags look the same
                      set.seed(y)

                      cv_folds <- cv_splits_all |>
                        dplyr::filter(.data$common_seed == y) |>
                        purrr::pluck('cv_splits', 1)

                      return(
                        list(
                          recipe = x,
                          cv_folds = cv_folds,
                          seed = y,
                          bag = .bag)
                      )
                    })

  return(out)
}
