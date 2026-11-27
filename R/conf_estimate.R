#' Compute a confidence level estimate
#'
#' @description Computing confidence level estimates for the predicted class
#' based on a beta distribution fitted on the random forest scores and
#' computing the probability mass below or above the classification threshold,
#' if the class is 0 or 1, respectively
#'
#' @param scores numeric vector with all the random forest scores of one
#' vessel-year
#' @param pred_class predicted class (0 or 1) of that vessel-year
#' @param threshold threshold to classify the scores into 0 or 1
#'
#' @return confidence level estimate, a single number
#'

conf_estimate <- function(scores, pred_class, threshold) {

  if ((length(scores) > 1 &&
       (all(scores == 1) || all(scores == 0))) ||
      length(unique(scores)) == 1) {
    conf <- 1
  } else {
    # beta fitting; scores of exactly 0 or 1, or NA, make ebeta() warn
    beta_par <- suppressWarnings(
      EnvStats::ebeta(scores, method = "mle")$parameters
    )

    if (pred_class == 1) {
      conf <- stats::pbeta(q = threshold,
                           shape1 = beta_par[1],
                           shape2 = beta_par[2],
                           lower.tail = FALSE)
    } else {
      conf <- stats::pbeta(q = threshold,
                           shape1 = beta_par[1],
                           shape2 = beta_par[2],
                           lower.tail = TRUE)
    }
  }
  return(conf)
}
