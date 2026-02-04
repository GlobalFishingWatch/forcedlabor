#' Compute alpha*_n from DEDPUL
#'
#' @description Computing alpha star, or the upper bound of alpha, the
#' proportion of positives within the unlabeled (populationwise). Function
#' used in dedpul_estimation
#'
#' @param r sorted 1D array of density ratios
#' @param steps Number of locations at which to compute D
#' @param plotting if TRUE, a D vs. alpha plot is generated
#' @param filename if plotting is TRUE, a filename with path is required
#'
#' @return estimated alpha*_n
#'
#' @details We first compute D; see reference
#' The nominal shape of an alpha* vs D plot is:
#'     |          /
#'     |         /
#'     |        /
#' D=0 + ------/
#'     |
#'     +------------------------
#'               alpha
#'
#' alpha* is located at the corner where D departs from zero.
#' DEDPUL uses two approaches to find this location. One is to find
#' the rightmost point where D is zero, the resulting estimate is
#' referred to as alpha*_c. This is computed using the EM algorithm
#' in DEDPUL, but it seems to be unstable and can get stuck in very small
#' values of alpha, so we're not computing it.
#'
#' The second approach is to find the maximum of the second derivative of
#' D. This is referred to as alpha*_n and we compute it exactly as it
#' is computed in the DEDPUL paper (Algorithm 2).
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @import ggplot2
#'
#'

compute_alpha_star <- function(r, steps = 1000, plotting = FALSE,
                               filename = NULL) {

  D_alpha <- compute_D(r, steps)

  D2 <- data.frame(alpha = D_alpha$alpha[2:(nrow(D_alpha) - 1)],
                   D_2 = D_alpha$D[3:nrow(D_alpha)] +
                     D_alpha$D[1:(nrow(D_alpha) - 2)] -
                     2 * D_alpha$D[2:(nrow(D_alpha) - 1)])

  alpha_n <- D2$alpha[which.max(D2$D_2)]

  if (plotting == TRUE & is.null(filename) == FALSE) {
    ggplot2::ggplot(data = D_alpha, ggplot2::aes(x = .data$alpha, y = .data$D)) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::geom_point(ggplot2::aes(x = alpha_n, y =
                                D_alpha$D[which.max(D2$D_2) + 1]),
                          size = 4, shape = 22, fill = "black") +
      ggplot2::theme_bw()
    ggplot2::ggsave(filename = filename)
  }

  return(alpha_n)
}
