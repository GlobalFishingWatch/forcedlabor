#' Enforcing partial monotonicity on r
#'
#' @description Each element of r is forced to be monotonic where, for the
#' element, y_u > y_u.mean(). See Algorithm 2 in reference for more details.
#'
#' @param r sorted array of density ratios
#' @param y_u vector with the predictions of unlabeled
#' @return sorted array of density ratios, monotonized
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'

monotonize <- function(r, y_u) {

  threshold_mon <- mean(y_u)

  max_r <- 0

  for (i in seq_along(r)) {
    if (y_u[i] > threshold_mon) {
      max_r <- max(r[i], max_r)
      r[i] <- max_r
    }
  }

  return(r)
}
