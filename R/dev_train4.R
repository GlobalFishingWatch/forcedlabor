#' Training forced labor random forest model.
#'
#' @param cv_setup List containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers. Output from ?cv_setup
#' @param rf_spec Random forest classifier specifications
#' @param free_cores Number of available cores. Add more if you need to do many things at the same time
#' @param parallel_plan Parallelization strategy Options: multisession (if running RStudio), multicore (Linux, Mac and plain R) or psock (if multisession is not working well and you need to try something else)
#' @param save_dir Directory to save trained models otherwise skip saving when NULL
#'
#' @returns List containing:
#' Trained random forest models
#' Tibble with predicted probabilities
#'
#' @importFrom dplyr bind_cols bind_rows mutate select
#' @importFrom furrr future_pmap furrr_options
#' @importFrom future cluster multicore multisession plan
#' @importFrom parallel detectCores stopCluster
#' @importFrom parallelly availableCores makeClusterPSOCK
#' @importFrom purrr map pmap
#' @import workflows
#'
#' @export
dev_ml_train4 <- function(x = tlist[[1]],
                          rf_spec = rf_setup$rf_spec,
                          new_data = NULL,
                          save_dir = NULL) {
   #bag, recipe, folds_tbl) {
      seed <- x$seed
      bag <- x$bag
      recipe <- x$recipe
      folds_tbl <- x$folds_tbl
      workflow <- workflows::workflow() |>
        workflows::add_model(rf_spec) |>
        workflows::add_recipe(recipe)

      out2 <- purrr::pmap(
        list(folds_tbl$splits, seed, folds_tbl$id, bag),
        \(x, y, z, b) {
          set.seed(y)
          ind_anal   <- rsample::analysis(x)
          ind_assess <- rsample::assessment(x)
          # fit the model
          tmp_model <- workflows:::fit.workflow(workflow, ind_anal)
          #saving if provided

          if (!is.null(save_dir)) {
            dir.create(save_dir, showWarnings = F)
            # model_id (for loading) and file path
            model_id <- paste0("rf_seed", y, "_bag", b, "_", z)
            print(paste("saving", model_id))
            file_path <- file.path(save_dir, paste0(model_id, ".rds"))
            saveRDS(tmp_model, file_path)
          } else {
            print("Skipping model saving")
          }
          # prediction ind_assess
          tmp_pred_assess <- workflows:::predict.workflow(
            object = tmp_model,
            new_data = ind_assess,
            type = "prob"
          ) |>
            dplyr::select(.pred_1) |>
            dplyr::bind_cols(ind_assess[c("indID","known_offender","known_non_offender")]) |>
            dplyr::mutate(
              holdout = 0,
              common_seed = y,
              bag = b,
              id = z
            )
          # prediction new_data
          if (!is.null(new_data)) {
            tmp_pred <- workflows:::predict.workflow(object = tmp_model,
                                                     new_data = new_data,
                                                     type = "prob") |>
              dplyr::select(.pred_1) |>
              # Add columns to prediction data
              # (might be a warning about levels in source_id but it's not important,
              # we won't use that column anyway)
              dplyr::bind_cols(new_data[c("indID", "known_offender", "known_non_offender")]) |>
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

