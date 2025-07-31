#' Computes binary classification via DEDPUL
#'
#' @description For each vessel-year, it computes a binary
#' classification, 0 non offender and 1 offender. It is based on the DEDPUL
#' algorithm in the reference.
#'
#' @param data tibble with at least a common_seed column and a prediction_output
#' column The prediction_output column is a list. Each element contains a
#' tibble with predictions and covariates.
#' @param steps number of locations at which to compute D
#' @param plotting if TRUE, a D vs. alpha plot is generated
#' @param filepath if plotting is TRUE, a filepath of where to save the plot is
#' needed
#' @param threshold potential thresholds to test
#' @param eps accepted difference (tolerance) between alpha and the actual
#' proportion of positives for a given threshold
#' @param parallel_plan type of parallelization to run (multicore, multisession,
#' or psock - this last one may need calling libraries inside)
#' @param free_cores number of free cores to leave out of parallelization
#' @return tibble with classification and calibrated threshold used for them
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @importFrom furrr future_map
#' @importFrom future plan
#' @importFrom parallel detectCores
#' @importFrom parallel stopCluster
#' @importFrom parallelly makeClusterPSOCK
#' @importFrom parallelly availableCores
#' @importFrom purrr map2_dbl
#' @importFrom EnvStats ebeta
#' @importFrom stats pbeta
#' @import dplyr
#'
#' @export
#'

ml_classification <- function(data,
                              steps = 1000,
                              plotting = FALSE,
                              filepath = NULL,
                              threshold = seq(0, .99, by = 0.01),
                              eps = 0.01,
                              parallel_plan = "multicore",
                              free_cores = 1) {


  # first, checking if a good file name has been provided (the path exists)
  # only if plotting is TRUE

  if (plotting == TRUE) {
    if (dir.exists(filepath) == FALSE)
      stop("The directory to save the plot does not exist.")
  }

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
  # options(future.globals.maxSize = 1000000000)

  # unnesting the tibble inside the tibble
  # scores_df <- data |>
    # dplyr::select(.data$common_seed, .data$predictions) |>
    # tidyr::unnest(.data$predictions) # |>  # from having a list per cell to
    # # a tibble per cell
    # tidyr::unnest(.data$predictions)

  avgscore_df <- data |>
    # dplyr::select(.data$predictions) |>
    # tidyr::unnest(.data$predictions) |>  # from having a list per cell to
    # a tibble per cell
    # tidyr::unnest(.data$prediction_output) |> # everything is a regular tibble
    dplyr::group_by(dplyr::across(c(.data$indID,
                                    .data$holdout,
                                    .data$known_offender,
                                    .data$known_non_offender))) |>  # group by everything
    # except .pred_1 (only common_seed and indID actually matter but the other
    # don't make a diff in the calculations and it's useful to have them for
    # later)
    dplyr::summarize(pred_mean = mean(.data$.pred_1, na.rm = TRUE),
                     .groups = "drop")

  avgscore_df_noneg <- avgscore_df |>
    dplyr::filter(.data$holdout == 0)

  # getting a calibrated threshold based on the dedpul algorithm

  if (plotting == TRUE) {
    filename <- paste0(filepath, paste0("D_alpha_common_seed.png"))
  } else {
    filename <- NULL
  }


  threshold_res <- calibrated_threshold(data = avgscore_df_noneg,
                                        steps = steps,
                                        plotting = plotting,
                                        filename = filename,
                                        threshold = threshold,
                                        eps = eps)

  # classification
  predclass_df <- avgscore_df |>
    dplyr::mutate(pred_class = purrr::map2_dbl(.data$pred_mean,
                                               threshold_res$thres_star, function(x, y) {
                                                 ifelse(x > y, 1, 0)}))

  if (length(unique(data$common_seed)) + length(unique(data$bag)) > 2) {

    confidence <- t( do.call(
      cbind.data.frame,parallel::mclapply( split(predclass_df, predclass_df$indID),
              FUN = conf_estimate, data = data,
              threshold = threshold_res$thres_star,
              mc.cores = detectCores() - free_cores)))

    predclass_df$conf <- c(confidence)



  } else {
    print('not enough data to compute confidence levels')

  }


  return(list(pred_conf = predclass_df,
              alpha = threshold_res$alpha,
              threshold = threshold_res$thres_star))
}
