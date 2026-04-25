#' Loading trained rf models from disk based on the cross validation set up defined
#'
#' @param cv_setup List containing cv_folds (analysis/assessment), model workflow and seed/bag identifiers. Output from ?cv_setup
#' @param save_dir Directory to save trained models otherwise skip saving when NULL
#'
#' @returns List containing:
#' Trained random forest models
#'
#' @importFrom dplyr bind_rows
#' @importFrom purrr map pmap
#' @importFrom tibble tibble
#'
#'

fl_load <- function(cv_setup,
                    rf_spec = rf_setup$rf_spec,
                    save_dir) {
list.files(save_dir)
  # Check directory exists
  if (!dir.exists(save_dir)) {
    stop("Directory does not exist: ", save_dir)
  }

  # Load models matching the cv_setup structure
  out <- purrr::pmap(
    list(
      purrr::map(cv_setup, "seed"),
      purrr::map(cv_setup, "bag"),
      purrr::map(cv_setup, "cv_folds")
    ),
    function(seed, bag, folds_tbl) {

      # Load models for each fold in this seed/bag combination
      models_list <- purrr::map(
        folds_tbl$id,
        function(fold_id) {

          # Reconstruct the model_id and file_path
          model_id <- paste0("rf_seed", seed, "_bag", bag, "_", fold_id)
          file_path <- file.path(save_dir, paste0(model_id, ".qs2"))
          file_exists <- file.exists(file_path)

          # Stop if missing and fail_on_missing is TRUE
          if (!file_exists) {
            stop("Required model file not found: ", file_path)
          } else {
            model <- qs2::qs_read(file_path)
          }

          # Return in the same format as training
          list(
            model = tibble::tibble(
              model_id = model_id,
              seed = seed,
              bag = bag,
              fold_id = fold_id,
              model_object = list(model)
            )
          )
        }
      )

      # Bind rows for this workflow/seed/bag (matching training structure)
      return(list(
        models = dplyr::bind_rows(purrr::map(models_list, "model"))
      ))
    }
  )

  return(purrr::map(out, "models"))
}
