#' Apply downsample step to data recipe across defined seeds/bags
#'
#' @param bag_runs Tibble defining bag numbers and seeds
#' @param fl_rec Model recipe
#' @param down_sample_ratio See `under_ratio` [themis::step_downsample()].
#' Downsampling ratio to balance the number of positive and unlabelled cases.
#' Defaults to 1 (1:1 ratio)
#'
#' @returns Tibble with data recipe and downsample ratio.
#'
#' @importFrom dplyr mutate
#' @importFrom purrr map
#' @importFrom themis step_downsample

bag_downsample <- function(bag_runs,
                           fl_rec,
                           down_sample_ratio) {
  bag_runs |>
    dplyr::mutate(
      fl_recipe = purrr::map(.data$recipe_seed, function(x) {
        fl_rec_down <- fl_rec |>
          themis::step_downsample(known_offender,
                                  under_ratio = down_sample_ratio,
                                  seed = x,
                                  skip = TRUE)
      })
    )
}
