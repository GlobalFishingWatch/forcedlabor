
#' Computes prediction (classification) summary between seeds
#'
#' @description The mode of the class (0 or 1) is computed between seeds, and
#' the proportion of seeds matching that classification is given too.
#'
#' @param data tibble with at least an indID (vessel-year ID) column, pred_class
#' column (predicted class in each seed),  thres column (threshold used for
#' classification with the corresponding common seed), and common_seed column.
#' @param num_common_seeds number of common seeds
#' @return tibble with recall and specificity per seed
#'
#' @import dplyr
#'
#' @export
#'

ml_pred_summary <- function(data, num_common_seeds) {

  pred_class_stats <- data %>%
    dplyr::group_by(.data$indID) %>%
    dplyr::add_count(.data$pred_class, sort = TRUE) %>%
    dplyr::slice(1) %>% # we're assuming that there will be no ties
    dplyr::mutate(class_mode = .data$pred_class,
                  class_prop = n / num_common_seeds) %>%
    dplyr::select(-c(.data$pred_class, n, .data$thres, .data$common_seed))

  return(pred_class_stats)
}
