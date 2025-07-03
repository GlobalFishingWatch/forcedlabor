#' Compute D from MAX_SLOPE algorithm
#'
#' @details See Algorithm 2 in the reference.
#' \code{D = alpha - mean(p(Yu))} where \code{p(Yu)} is defined as
#' \code{p(Yu) = min(alpha * r(Yu), 1)}
#'
#' @param r sorted 1D array of density ratios
#' @param steps number of locations at which to compute D
#' @return alpha and D vectors
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#'

compute_D <- function(r, steps = 1000) {

  alpha_vector <- seq(from = 0, to = 1, length.out = steps)
  alpha_py <- as.matrix(r) %*% t(as.matrix(alpha_vector))
  alpha_py_min <- pmin(alpha_py, 1)
  D_alpha <- data.frame(alpha = alpha_vector,
                        D = alpha_vector - apply(alpha_py_min, 2, mean))

  return(D_alpha)

}
