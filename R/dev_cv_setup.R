#' Defining cross-validation folds (analysis/assessment) and model workflows across bags.
#'
#' @param bag_runs Tibble defining bag numbers and seeds
#' @param cv_splits_all Cross-validation data splits
#' @param fl_rec Recipes data recipe
#' @param rf_spec Random forest model specification
#' @param down_sample_ratio See under_ratio in ?themis::step_downsample
#'
#' @returns List object containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers
#'
#' @importFrom dplyr filter mutate select
#' @importFrom purrr map pluck
#' @importFrom themis step_downsample
#' @importFrom workflows add_model add_recipe workflow
#'
#' @export
#'
#' @examples

dev_cv_setup <- function(bag_runs,
                     cv_splits_all,
                     fl_rec,
                     rf_spec,
                     down_sample_ratio)
{

  down_bags<-dev_bag_downsample(bag_runs = bag_runs,
                            fl_rec = fl_rec,
                            down_sample_ratio = down_sample_ratio)

  out<-purrr::pmap(list(down_bags$fl_recipe,
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
                       workflows::add_model(rf_spec) |>
                       workflows::add_recipe(x)

                     # get the folds related to that common seed, train and predict
                     cv_predictions <-
                       cv_splits_all |>
                       dplyr::filter(common_seed == y) |>
                       purrr::pluck('cv_splits',1) |> # unlist first (unique) element
                       dplyr::mutate(# Create analysis dataset based on CV folds
                         analysis = purrr::map(splits, ~rsample::analysis(.x)),
                         # Create assessment dataset based on CV folds
                         assessment = purrr::map(splits, ~rsample::assessment(.x))) |>
                       dplyr::select(-splits)

                     return(
                       list(
                         workflow   = cv_predictions_workflow,
                         cv_folds = cv_predictions,
                         seed = y,
                         bag = .bag
                       )
                     )

                   })

  return(out)
}
