#' Computes density ratios array r
#'
#' @description Compute r using Algorithms 1 and 2 in reference.
#'
#' @param f_y list with f_yp, f_yu and y_u as elements (see details)
#' @return sorted array of density ratios, monotonized and smoothed
#'
#' @details In f_y, f_yp is the vector of inferred densities for positive
#' predictions, f_yu is the vector of inferred densities for unlabeled
#' predictions, and y_u is the vector with the predictions of unlabeled.
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#'

compute_r <- function(f_y) {

  r <- f_y$f_yp / f_y$f_yu

  # monotonizing
  r <- monotonize(r = r, y_u = f_y$y_u)

  # rolling median
  r <- rolling_median(r)

  return(r)
}
