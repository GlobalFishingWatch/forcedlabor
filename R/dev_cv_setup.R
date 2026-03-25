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
                          down_sample_ratio)
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

  out<-purrr:::pmap(list(down_bags$fl_recipe,
                         down_bags$common_seed,
                         down_bags$bag), # previously future_map2, now pmap to map 3 inputs
                    function(x, y, .bag) # added .data$bag as third mapped input
                    {

                      # Ensure all bags look the same
                      set.seed(y)

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
                          bag = .bag)
                      )
                    })

  return(out)
}
