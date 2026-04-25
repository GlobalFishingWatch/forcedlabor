#' Smoothing r using a rolling median
#'
#' @param r sorted array of density ratios
#' @param l_2 denominator to get rolling window of length(r)/l_2
#' (default to 20 based on the reference)
#' @return sorted array of density ratios, smoothed
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @importFrom stats runmed
#'
#'

rolling_median <- function(r, l_2 = 20) {

  r <-  stats::runmed(r, k = length(r) / l_2, algorithm = "Turlach",
                      na.action = "na.omit")

  return(r)
}
