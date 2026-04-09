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
#'

dev_ml_train2 <- function(cv_setup,
                          rf_spec,
                         free_cores,
                         parallel_plan,
                         save_dir = NULL) {

  # Setting up the parallelization
  if (parallel_plan == "multicore") {
    future::plan(future::multicore,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  } else if (parallel_plan == "psock") {
    cl <- parallelly::makeClusterPSOCK(parallelly::availableCores() - free_cores)
    future::plan(future::cluster, workers = cl)
  } else {
    future::plan(future::multisession,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  # creating directory if provided
  if (!is.null(save_dir)) {
    if (dir.exists(save_dir) == FALSE) {
      dir.create(save_dir)
    }
  } else{
    print("Skipping model saving")
  }

  out <- furrr::future_pmap(
      list(
      purrr::map(cv_setup, "seed"),
      purrr::map(cv_setup, "bag"),
      purrr::map(cv_setup, "recipe"),
      purrr::map(cv_setup, "cv_folds")
    ),
    .options = furrr::furrr_options(seed = TRUE, chunk_size = 1),
    .f = function(seed, bag, recipe, folds_tbl) {
      # Ensure all bags look the same
      set.seed(seed)

      workflow <- workflows::workflow() |>
        workflows::add_model(rf_spec) |>
        workflows::add_recipe(recipe)

      out_2 <- purrr::pmap(
        list(
          folds_tbl$splits,
          folds_tbl$id
          ),
        function(split, fold_id) {
          # reproducible within each outer future worker
          set.seed(seed)

          ind_anal   <- rsample::analysis(split)
          ind_assess <- rsample::assessment(split)

          # fit the model
          tmp_model <- workflows:::fit.workflow(workflow, ind_anal)

          # model_id (for loading) and file path
          model_id <- paste0("rf_seed", seed, "_bag", bag, "_", fold_id)
          file_path <- if (!is.null(save_dir)) {
            file.path(save_dir, paste0(model_id, ".rds"))
          } else {
            NA_character_
          }

          if (!is.null(save_dir)) {
            saveRDS(tmp_model, file_path)
          }

          tmp_pred_assess <- workflows:::predict.workflow(
            object = tmp_model,
            new_data = ind_assess,
            type = "prob"
          ) |>
            dplyr::select(.data$.pred_1) |>
            dplyr::bind_cols(ind_assess[c("indID","known_offender","known_non_offender")]) |>
            dplyr::mutate(
              holdout = 0,
              common_seed = seed,
              bag = bag,
              id = fold_id
            )

          #GM: to free RAM
          rm(ind_anal, ind_assess, tmp_model)
          gc()

          model_info = tibble::tibble(
            model_id = model_id,
            seed = seed,
            bag = bag,
            fold_id = fold_id,
            file_path = file_path,
            saved_to_disk = !is.null(save_dir),
            #model_object = list(tmp_model)
          )

          return(list(
            model = model_info,
            pred_assess = tmp_pred_assess
          ))
        }
      )

      return(list(
        pred_assess = purrr::map(out_2, "pred_assess")
      ))
    }
  )

  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }

  #list(#GM: comment out
    #fitted_models = purrr::map(out, "models"),
    train_probabilities = purrr::map(out, "pred_assess")#dplyr::bind_rows() #GM: to remove dplyr::bind_rows
  #)#GM: comment out
}
