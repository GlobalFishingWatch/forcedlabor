#' Training forced labor random forest model.
#'
#' @param cv_setup List containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers. Output from ?cv_setup
#' @param free_cores Number of available cores. Add more if you need to do many things at the same time
#' @param parallel_plan Parallelization strategy Options: multisession (if running RStudio), multicore (Linux, Mac and plain R) or psock (if multisession is not working well and you need to try something else)
#' @param save_dir Directory to save trained models
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

dev_ml_train <- function(cv_setup,
                         free_cores,
                         parallel_plan,
                         save_dir){

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

  # creating directory
  if (dir.exists(save_dir) == FALSE){
    dir.create(save_dir)
  }

  out <- furrr::future_pmap(
    list(
      purrr::map(cv_setup, "workflow"),
      purrr::map(cv_setup, "seed"),
      purrr::map(cv_setup, "bag"),
      purrr::map(cv_setup, "cv_folds")
    ),
    function(workflow, seed, bag, folds_tbl){

      # if (!"themis" %in% loadedNamespaces())
      #   requireNamespace("themis", quietly = TRUE)

      out_2 <- purrr::pmap(
        list(
          folds_tbl$analysis,
          folds_tbl$assessment,
          folds_tbl$id
        ),
        function(ind_anal, ind_assess, fold_id){

          # reproducible within each outer future worker
          set.seed(seed)

          # fit the model
          tmp_model <- workflows:::fit.workflow(workflow, ind_anal)

          # save model
          file_name <- file.path(
            save_dir,
            paste0("rf_seed", seed, "_bag", bag, "_", fold_id, ".rds")
          )
          saveRDS(tmp_model, file_name)

          tmp_pred_assess <- workflows:::predict.workflow(
            object = tmp_model,
            new_data = ind_assess,
            type = "prob"
          ) |>
            dplyr::select(.pred_1) |>
            dplyr::bind_cols(ind_assess[c("indID","known_offender","known_non_offender")]) |>
            dplyr::mutate(
              holdout = 0,
              common_seed = seed,
              bag = bag,
              id = fold_id
            )

          return(list(
            model = tmp_model,
            pred_assess = tmp_pred_assess
          ))
        }
      )

      # return aggregated outputs for this workflow/seed/bag
      return(list(
        models = purrr::map(out_2, "model"),
        pred_assess = dplyr::bind_rows(purrr::map(out_2, "pred_assess"))
      ))
    },
    .options = furrr::furrr_options(seed = TRUE, packages = c("themis"))
  )

  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }

  #GM: currently saving as list objects, would you prefer models to be stored in a tibble?
  list(
    fitted_models = purrr::map(out, "models"),
    train_probabilities = dplyr::bind_rows(purrr::map(out, "pred_assess"))
  )
}
