#' Training forced labor random forest model.
#'
#' @param cv_setup List containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers. Output from ?cv_setup
#' @param rf_spec Random forest classifier specifications
#' @param save_dir Directory to save trained models otherwise skip saving when NULL
#' @param holdout Optional. Test dataset, not used for model training.
#'
#' @returns List containing:
#' Trained random forest models
#' Tibble with predicted probabilities
#'
#' @importFrom dplyr bind_cols bind_rows mutate select
#' @importFrom purrr map pmap
#' @importFrom qs2 qs_save
#' @import workflows
#'
#' @export
ml_train <- function(cv_setup = cv_df[[1]],
                     rf_spec = rf_setup$rf_spec,
                     holdout = NULL,
                     save_dir = NULL) {
  seed <- cv_setup$seed
  bag <- cv_setup$bag
  recipe <- cv_setup$recipe
  folds_tbl <- cv_setup$cv_folds
  workflow <- workflows::workflow() |>
    workflows::add_model(rf_spec) |>
    workflows::add_recipe(recipe)
  out2 <- purrr::pmap(
    list(folds_tbl$splits, seed, folds_tbl$id, bag), \(x, y, z, b) {
      set.seed(y)
      ind_anal   <- rsample::analysis(x)
      ind_assess <- rsample::assessment(x)
      # fit the model
      tmp_model <- workflows:::fit.workflow(workflow, ind_anal)
      #saving if provided
      if (!is.null(save_dir)) {
        dir.create(save_dir, showWarnings = FALSE)
        # model_id (for loading) and file path
        model_id <- paste0("rf_seed", y, "_bag", b, "_", z)
        print(paste("saving", model_id))
        file_path <- file.path(save_dir, paste0(model_id, ".qs2"))
        qs2::qs_save(tmp_model, file_path)
      }
      # prediction ind_assess
      tmp_pred_assess <- workflows:::predict.workflow(
        object = tmp_model,
        new_data = ind_assess,
        type = "prob" ) |>
        dplyr::select(.pred_1) |>
        dplyr::bind_cols(ind_assess[c("indID", "known_offender",
                                      "known_non_offender")]) |>
        dplyr::mutate(
          holdout = 0,
          common_seed = y,
          bag = b,
          id = z
        )
      # prediction holdout
      if (!is.null(holdout)) {
        tmp_pred <- workflows:::predict.workflow(object = tmp_model,
                                                 new_data = holdout,
                                                 type = "prob") |>
          dplyr::select(.pred_1) |>
          # Add columns to prediction data
          dplyr::bind_cols(holdout[c("indID", "known_offender",
                                      "known_non_offender")]) |>
          dplyr::mutate(holdout = 1,
                        common_seed = b,
                        bag = b,
                        id = z)
      }
      rm(ind_anal, ind_assess, tmp_model)
      gc()

      return(list(
        pred_assess = tmp_pred_assess,
        pred_new = get0("tmp_pred")
      ))
    }
      )
  return(list(
    train_probabilities = purrr::map(out2, "pred_assess"),
    pred_probabilities = purrr::map(out2, "pred_new"))
  )
  }

