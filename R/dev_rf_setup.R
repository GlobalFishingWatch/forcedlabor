#' Setting up the model recipe and random forest specification
#'
#' @param training_data A tibble with training data
#' @param y Character, name of the response variable. Defaults to `"known_offender"`
#' @param x Vector with names of the features to be used. Defaults to NULL and should be set accordingly to avoid adding new features without knowing
#' @param id ID variable name
#' @param dont_use Exclude a given variable (or variables) as predictors
#' @param control Control variable column
#' @param corr_threshold A value for the threshold of absolute correlation values. The step will try to remove the
#' minimum number of columns so that all the resulting absolute correlations are less than this value. See ?recipes::step_corr for further information
#' @param rf_trees Number of trees contained in the ensemble. The larger the number of trees,
#' the more stable the predictions - but with a higher computational cost
#' @param rf_mtry Number of features that will be sampled at each split
#' @param rf_min_n Minimum node size (i.e. the number of vessel-years that a final node should have).
#' Larger values create less complex trees, and may result in underfitting. Smaller values create more complex trees, and may result in overfitting
#' @param rf_reg_factor Regularization factor works by penalizing new variables by multiplying the splitting criterion by
#' a factor. As a result, the tree is biased to keep splitting on features it already trusts, rather than constantly introducing fresh ones
#'
#' @returns List containing random forest recipe and specifications
#'
#' @importFrom parsnip rand_forest set_engine set_mode
#' @importFrom recipes recipe step_corr step_nzv update_role
#'
#' @export

dev_rf_setup <- function(training_data,
                         y = "known_offender", #response
                         x = NULL,
                         id = "indID",
                         dont_use = c("flag_region", "known_non_offender"),
                         control = "source_id_number",
                         corr_threshold = 0.75,
                         rf_trees = 500,
                         rf_mtry = 1,
                         rf_min_n = 15,
                         rf_reg_factor = 0.5) {

  rf_recipe <-
    recipes::recipe(training_data) |>
    recipes::update_role(y,
                         new_role = "outcome") |>
    recipes::update_role(tidyselect::all_of(x),
                         new_role = "predictor") |>
    recipes::update_role(tidyselect::all_of(id),,
                         new_role = "id") |>
    recipes::update_role(tidyselect::all_of(dont_use),
                         new_role = "dont_use")  |>
    recipes::update_role(tidyselect::all_of(control), new_role = "control")  |>
    recipes::step_nzv(recipes::all_predictors())  |>
    recipes::step_corr(recipes::all_numeric(), threshold = corr_threshold)

  rf_spec <-
    parsnip::rand_forest(trees = rf_trees,
                         mtry = rf_mtry,
                         min_n = rf_min_n) |>
    parsnip::set_mode("classification") |>
    parsnip::set_engine("ranger",
                        regularization.factor = rf_reg_factor)

  return(list(rf_recipe = rf_recipe,
              rf_spec = rf_spec))

}
