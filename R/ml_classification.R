#' Computes binary classification via DEDPUL
#'
#' @description For each vessel-year, it computes a binary
#' classification, 0 non offender and 1 offender. It is based on the DEDPUL
#' algorithm in the reference.
#'
#' @param data tibble with at least a common_seed column and a prediction_output
#' column The prediction_output column is a list. Each element contains a
#' tibble with predictions and covariates.
#' @param common_seed_tibble tibble with one column containing all the common
#' seeds
#' @param steps number of locations at which to compute D
#' @param plotting if TRUE, a D vs. alpha plot is generated
#' @param filepath if plotting is TRUE, a filepath of where to save the plot is
#' needed
#' @param threshold potential thresholds to test
#' @param eps accepted difference (tolerance) between alpha and the actual
#' proportion of positives for a given threshold
#' @return tibble with classification and calibrated threshold used for them
#'
#' @references
#'
#' D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
#' Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
#' Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
#' 10.1109/ICMLA51294.2020.00128.
#'
#' @importFrom purrr map_dbl
#' @importFrom purrr map2_dbl
#' @importFrom EnvStats ebeta
#' @importFrom stats pbeta
#' @import dplyr
#'
#' @export
#'

ml_classification <- function(data, common_seed_tibble, steps = 1000,
                              plotting = FALSE, filepath = NULL,
                              threshold = seq(0, .99, by = 0.01), eps = 0.01) {


  # first, checking if a good file name has been provided (the path exists)
  # only if plotting is TRUE

  if (plotting == TRUE) {
    if (dir.exists(filepath) == FALSE)
      stop("The directory to save the plot does not exist.")
  }

  # unnesting the tibble inside the tibble
  scores_df <- data %>%
    dplyr::select(.data$common_seed, .data$prediction_output) %>%
    tidyr::unnest(.data$prediction_output) %>% # from having a list per cell to
    # a tibble per cell
    tidyr::unnest(.data$prediction_output)

  avgscore_df <- data %>%
    dplyr::select(.data$prediction_output) %>%
    tidyr::unnest(.data$prediction_output) %>% # from having a list per cell to
    # a tibble per cell
    tidyr::unnest(.data$prediction_output) %>% # everything is a regular tibble
    dplyr::group_by(dplyr::across(-.data$.pred_1)) %>% # group by everything
    # except .pred_1 (only common_seed and indID actually matter but the other
    # don't make a diff in the calculations and it's useful to have them for
    # later)
    dplyr::summarize(pred_mean = mean(.data$.pred_1, na.rm = TRUE),
                     .groups = "drop")

  avgscore_df_noneg <- avgscore_df %>%
    dplyr::filter(.data$holdout == 0)

  # getting a calibrated threshold based on the dedpul algorithm

  if (plotting == TRUE) {
    filename <- paste0(filepath, paste0("D_alpha_common_seed.png"))
  }else{
    filename <- NULL
  }


  threshold_res <- calibrated_threshold(data = avgscore_df_noneg, steps = steps,
                                        plotting = plotting,
                                        filename = filename,
                                        threshold = threshold,
                                        eps = eps)

  # classification
  predclass_df <- avgscore_df %>%
    dplyr::mutate(pred_class = purrr::map2_dbl(.data$pred_mean,
                                               threshold_res, function(x, y) {
                                                 ifelse(x > y, 1, 0)}))


  pred_conf <- predclass_df |>
    dplyr::mutate(confidence = purrr::map_dbl(.data$indID, function(x){

      line_classif <- which(.data$indID == x) # in averaged data frame
      predictions <- scores_df$.pred_1[which(scores_df$indID == x)]

      if (length(predictions) > 1 && (all(predictions == 1) || all(predictions == 0))){
        conf <- 1
      }else{
        # beta fitting
        beta_par <- EnvStats::ebeta(predictions, method = "mle")$parameters

        if (.data$pred_class[line_classif] == 1){
          conf <- stats::pbeta(q = threshold_res,
                               shape1 = beta_par[1], shape2 = beta_par[2], lower.tail = FALSE)

        }else{
          conf <- stats::pbeta(q = threshold_res,
                               shape1 = beta_par[1], shape2 = beta_par[2], lower.tail = TRUE)
        }

      }

    }))

  return(pred_conf)

}


#' Computes average confidence scores
#'
#' @description For each common seed and vessel-year, it computes the average
#' confidence score.
#'
#' @param data tibble with at least a common_seed column and a prediction_output
#' column The prediction_output column is a list. Each element contains a
#' tibble with predictions and covariates.
#' @return an object (tibble) with all the covariates that identify the
#' vessel-years, and an average predicted probability for each vessel-year
#' and common seed
#'
#' @importFrom tidyr unnest
#' @import dplyr
#'
#' @export
#'

avg_confscore <- function(data) {

  confscore_df <- data %>%
    dplyr::select(.data$common_seed, .data$prediction_output) %>%
    tidyr::unnest(.data$prediction_output) %>% # from having a list per cell to
    # a tibble per cell
    tidyr::unnest(.data$prediction_output) %>% # everything is a regular tibble
    dplyr::group_by(dplyr::across(-.data$.pred_1)) %>% # group by everything
    # except .pred_1 (only common_seed and indID actually matter but the other
    # don't make a diff in the calculations and it's useful to have them for
    # later)
    dplyr::summarize(pred_mean = mean(.data$.pred_1, na.rm = TRUE),
                     .groups = "drop")

  return(confscore_df)

}



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

calibrated_threshold <- function(data, steps = 1000, plotting = FALSE,
                                 filename = NULL,
                                 threshold = seq(0, .99, by = 0.01),
                                 eps = 0.01) {

  # estimating alpha
  alpha <- dedpul_estimation(data, steps, plotting, filename)

  print(paste("alpha: ", alpha))

  # keep only the unlabeled
  data <- data %>%
    dplyr::filter(.data$known_offender == 0) %>%
    dplyr::select(.data$pred_mean)

  # recursively search for the optimal threshold
  for (i in rev(seq_len(length(threshold)))) {
    thres_star <- threshold[i]
    sum_pred <- sum(data$pred_mean > thres_star)
    if (abs(sum_pred / dim(data)[1] - alpha) < eps) {
      break
    }
  }

  return(thres_star)

}


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

dedpul_estimation <- function(data, steps = 1000, plotting = FALSE,
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


#' Computes density kernels estimated for positive and unlabeled, and the values
#' of those densities inferred for unlabeled predictions (sorted)
#'
#' @description Computes density kernels estimated for positive and unlabeled,
#' and the values of those densities inferred for unlabeled predictions (sorted)
#'
#' @param data data frame. Needs to have a .pred_1 column with predictions
#' and a known_offender column with 0 for unlabeled and 1 for positive
#' (offender)
#' @return a list with 3 elements:
#' f_yp : inferred densities for positive predictions;
#' f_yu : inferred densities for unlabeled predictions;
#' y_u  : vector with the predictions of unlabeled
#'
#' @importFrom KernSmooth bkde
#' @importFrom stats approx
#' @import dplyr
#'
#' @export
#'

kernel_unlabeled <- function(data) {

  # only predictions for offenders
  pred_pos <- data %>%
    dplyr::filter(.data$known_offender == 1) %>%
    dplyr::select(.data$pred_mean)
  # only predictions for unlabeled
  pred_unl <- data %>%
    dplyr::filter(.data$known_offender == 0) %>%
    dplyr::select(.data$pred_mean)
  # sorted predictions of unlabeled
  y_u <- sort(pred_unl$pred_mean)
  # density kernels and interpolation to the unlabeled values
  den_pos <- KernSmooth::bkde(pred_pos$pred_mean)
  den_pos_u <- stats::approx(den_pos$x, den_pos$y,
                             xout = sort(pred_unl$pred_mean), rule = 2)
  den_unl <- KernSmooth::bkde(pred_unl$pred_mean)
  den_unl_u <- stats::approx(den_unl$x, den_unl$y,
                             xout = sort(pred_unl$pred_mean), rule = 2)

  return(list(f_yp = den_pos_u$y, f_yu = den_unl_u$y, y_u = y_u))

}

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
#' @export
#'

compute_r <- function(f_y) {

  r <- f_y$f_yp / f_y$f_yu

  # monotonizing
  r <- monotonize(r = r, y_u = f_y$y_u)

  # rolling median
  r <- rolling_median(r)

  return(r)
}


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
#' @export

monotonize <- function(r, y_u) {

  threshold_mon <- mean(y_u)

  max_r <- 0

  for (i in seq_len(length(r))) {
    if (y_u[i] > threshold_mon) {
      max_r <- max(r[i], max_r)
      r[i] <- max_r
    }
  }

  return(r)
}


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
#' @export
#'

rolling_median <- function(r, l_2 = 20) {

  r <-  stats::runmed(r, k = length(r) / l_2, algorithm = "Turlach",
               na.action = "na.omit")

  return(r)
}


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
#' @export
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
    ggplot2::ggplot(data = D_alpha, aes(x = .data$alpha, y = .data$D)) +
      ggplot2::geom_line() +
      ggplot2::geom_point() +
      ggplot2::geom_point(aes(x = alpha_n, y =
                                D_alpha$D[which.max(D2$D_2) + 1]),
                          size = 4, shape = 22, fill = "black") +
      ggplot2::theme_bw()
    ggplot2::ggsave(filename = filename)
  }

  return(alpha_n)
}


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
#' @export
#'

compute_D <- function(r, steps = 1000) {

  alpha_vector <- seq(from = 0, to = 1, length.out = steps)
  alpha_py <- as.matrix(r) %*% t(as.matrix(alpha_vector))
  alpha_py_min <- pmin(alpha_py, 1)
  D_alpha <- data.frame(alpha = alpha_vector,
                        D = alpha_vector - apply(alpha_py_min, 2, mean))

  return(D_alpha)

}
