#' Computing the proportion of positives within the unlabeled (populationwise).
#'
#' @description Computes alpha star, or the upper bound of alpha, the proportion
#' of positives within the unlabeled (populationwise).
#'
#' @param data data frame. Needs to have a .pred_1 column with predictions
#' and a known_offender column with 0 for unlabeled and 1 for positive
#' (offender)
#' @param steps number of locations at which to compute D
#' @param plotting if TRUE, a D vs. alpha plot is created
#' @param filename if plotting is TRUE, a filename with path is required
#' @return estimated alpha value
#'
#' @details We first get density kernels estimated for positive and unlabeled
#' and the values of those densities inferred for unlabeled predictions
#' (sorted). Then we compute the sorted array of density
#' ratios. We finally use it to compute alpha star. The calculations are based
#' in the algorithm described in the reference.
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @export
#'

dedpul_estimation <- function(data,
                              steps = 1000,
                              plotting = FALSE,
                              filename = NULL) {

  # We get density kernels estimated for positive and unlabeled and the values
  # of those densities inferred for unlabeled predictions (sorted)
  f_y <- kernel_unlabeled(data)

  # Algorithms 1 and 2 in ref; r is the sorted array of density ratios
  r <- compute_r(f_y)

  # Computing alpha star, or the upper bound of alpha, the proportion of
  # positives within the unlabeled (populationwise)
  alpha <- compute_alpha_star(r, steps, plotting, filename)

  return(alpha)

}
