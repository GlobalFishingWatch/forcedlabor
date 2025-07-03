#' Computes threshold for offender classification
#'
#' @description Computes threshold for offender classification based on alpha
#' (from the dedpul_estimation function—see reference): which threshold would
#' achieve an alpha or proportion of positives within the unlabeled (more or
#' less) equal to alpha?
#'
#' @param data data frame. Needs to have a .pred_1 column with predictions
#' and a known_offender column with 0 for unlabeled and 1 for positive
#' (offender)
#' @param steps number of locations at which to compute D
#' @param plotting if TRUE, a D vs. alpha plot is generated
#' @param filename if plotting is TRUE, a filename with path is required
#' @param threshold potential thresholds to test
#' @param eps accepted difference (tolerance) between alpha and the actual
#' proportion of positives for a given threshold
#' @return a threshold to use
#'
#' @details For more details on the algorithm, please see the reference.
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @import dplyr
#'
#' @export
#'

calibrated_threshold <- function(data,
                                 steps = 1000,
                                 plotting = FALSE,
                                 filename = NULL,
                                 threshold = seq(0, .99, by = 0.01),
                                 eps = 0.01) {

  # estimating alpha
  alpha <- dedpul_estimation(data, steps, plotting, filename)

  print(paste("alpha: ", alpha))

  # keep only the unlabeled
  data <- data |>
    dplyr::filter(.data$known_offender == 0) |>
    dplyr::select(.data$pred_mean)

  # recursively search for the optimal threshold
  for (i in rev(seq_len(length(threshold)))) {
    thres_star <- threshold[i]
    sum_pred <- sum(data$pred_mean > thres_star)
    if (abs(sum_pred / dim(data)[1] - alpha) < eps) {
      break
    }
  }

  return(list(thres_star = thres_star, alpha = alpha))

}
