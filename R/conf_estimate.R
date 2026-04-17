#' Compute a confidence level estimate
#'
#' @description Computing confidence level estimates for the predicted class
#' based on a beta distribution fitted on the random forest scores and
#' computing the probability mass below or above the classification threshold,
#' if the class is 0 or 1, respectively
#'
#' @param predicted_df dataframe of one row containing the predicted class and
#' the ID of the vessel, that should match the IDs in data
#' @param data dataframe with all the random forest scores
#' @param threshold threshold to classify the scores into 0 or 1
#'
#' @return confidence level estimates
#'

conf_estimate <- function(predicted_df, data, threshold) {

  options(warn = - 1)
  predictions <- data$.pred_1[which(data$indID == predicted_df$indID)]
  if ((length(predictions) > 1 &&
       (all(predictions == 1) || all(predictions == 0))) ||
      length(unique(predictions)) == 1)  {
    conf <- 1
  } else {
    # beta fitting
    beta_par <- EnvStats::ebeta(predictions, method = "mle")$parameters

    if (predicted_df$pred_class == 1) {

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
