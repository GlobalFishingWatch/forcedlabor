# Apply downsample step to data recipe across defined seeds/bags

Apply downsample step to data recipe across defined seeds/bags

## Usage

``` r
bag_downsample(bag_runs, fl_rec, down_sample_ratio)
```

## Arguments

- bag_runs:

  Tibble defining bag numbers and seeds

- fl_rec:

  Model recipe

- down_sample_ratio:

  See `under_ratio`
  [`themis::step_downsample()`](https://themis.tidymodels.org/reference/step_downsample.html).
  Downsampling ratio to balance the number of positive and unlabelled
  cases. Defaults to 1 (1:1 ratio)

## Value

Tibble with data recipe and downsample ratio.
