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
#'

kernel_unlabeled <- function(data) {

  # only predictions for offenders
  pred_pos <- data |>
    dplyr::filter(.data$known_offender == 1) |>
    dplyr::select(.data$pred_mean)
  # only predictions for unlabeled
  pred_unl <- data |>
    dplyr::filter(.data$known_offender == 0) |>
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
