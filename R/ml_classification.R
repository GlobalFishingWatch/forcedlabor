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
#' @importFrom furrr future_map
#' @importFrom future plan
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

      split_df <- predclass_df |>
        split(predclass_df$indID)

      confidence_list <- furrr::future_map(
        .x = split_df,
        .f = \(x) conf_estimate(predicted_df = x,
                                data = data,
                                threshold = threshold_res$thres_star))

      count_null <- sum(lengths(confidence_list) == 0)

      if (count_null > 0) {
        print(paste0("nulls: ", count_null))
        confidence_list <- confidence_list[lengths(confidence_list) != 0]
      }

      confidence_vector <- t(do.call(cbind.data.frame, confidence_list))
      confidence_df <- data.frame(indID = rownames(confidence_vector),
                                  conf = confidence_vector)

      predclass_df <- dplyr::left_join(predclass_df, confidence_df, by = dplyr::join_by(indID))

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
