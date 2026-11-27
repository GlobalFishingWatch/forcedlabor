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
#' @param confidence_levels Boolean to compute confidence levels
#'
#' @return tibble with classification and calibrated threshold used for them
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @importFrom furrr future_map2_dbl
#' @importFrom future plan
#' @importFrom purrr map2_dbl
#' @importFrom EnvStats ebeta
#' @importFrom stats pbeta
#' @import dplyr
#'
#' @export
#'

fl_classification <- function(data,
                              steps = 1000,
                              plotting = FALSE,
                              filepath = NULL,
                              threshold = seq(0, .99, by = 0.01),
                              eps = 0.01,
                              confidence_levels = TRUE) {

  # first, checking if a good file name has been provided (the path exists)
  # only if plotting is TRUE

  if (plotting == TRUE) {
    if (dir.exists(filepath) == FALSE) dir.create(filepath, showWarnings = FALSE)
    filename <- paste0(filepath, paste0("D_alpha_common_seed.png"))
  } else {
    filename <- NULL
  }

  avgscore_df <- data |>
    dplyr::group_by(dplyr::across(c(indID,
                                    holdout,
                                    known_offender,
                                    known_non_offender))) |>
    dplyr::summarize(pred_mean = mean(.data$.pred_1, na.rm = TRUE),
                     .groups = "drop")

  avgscore_df_noneg <- avgscore_df |>
    dplyr::filter(holdout == 0)

  # getting a calibrated threshold based on the dedpul algorithm
  threshold_res <- calibrated_threshold(data = avgscore_df_noneg,
                                        steps = steps,
                                        plotting = plotting,
                                        filename = filename,
                                        threshold = threshold,
                                        eps = eps)

  # classification
  predclass_df <- avgscore_df |>
    dplyr::mutate(
      pred_class = purrr::map2_dbl(.data$pred_mean,
                                   threshold_res$thres_star,
                                   function(x, y) {ifelse(x > y, 1, 0)}))

  if (confidence_levels) {

    if (length(unique(data$common_seed)) + length(unique(data$bag)) > 2) {

      # one row per vessel-year is needed to match scores to predicted classes
      ids <- as.character(predclass_df$indID)
      bad_ids <- unique(ids[is.na(ids) | duplicated(ids)])
      if (length(bad_ids) > 0) {
        stop("Confidence levels need one row per vessel-year (indID). ",
             length(bad_ids), " indID value(s) are missing or appear in ",
             "more than one row, for example: ",
             paste(bad_ids[seq_len(min(5, length(bad_ids)))], collapse = ", "),
             call. = FALSE)
      }

      # group the scores once per vessel-year, in predclass_df row order
      scores_list <- split(data$.pred_1, data$indID)[ids]

      confidence_vector <- furrr::future_map2_dbl(
        .x = scores_list,
        .y = predclass_df$pred_class,
        .f = \(x, y) conf_estimate(scores = x,
                                   pred_class = y,
                                   threshold = threshold_res$thres_star))

      predclass_df$conf <- unname(confidence_vector)

    } else {
      message("Not enough data to compute confidence levels")

    }
  } else {
    message("Skipped confidence level estimation")
    }

  return(list(pred_conf = predclass_df,
              alpha = threshold_res$alpha,
              threshold = threshold_res$thres_star))
}
