# Setting up data structure

Sets up the data structure to train the RF model based on a number of
folds, bags and common seeds.

## Usage

``` r
fl_cvsetup(
  training_data,
  fl_rec,
  rf_spec,
  num_seeds,
  num_bags,
  num_folds,
  down_sample_ratio,
  group_var = "source_id"
)
```

## Arguments

- training_data:

  Dataset over which to generate CV folds and tuning grid

- fl_rec:

  Model recipe

- rf_spec:

  Random forest classifier specifications

- num_seeds:

  Number of common seeds

- num_bags:

  Number of bags

- num_folds:

  Number of cross validation folds

- down_sample_ratio:

  See `under_ratio`
  [`themis::step_downsample()`](https://themis.tidymodels.org/reference/step_downsample.html).
  Downsampling ratio to balance the number of positive and unlabelled
  cases. Defaults to 1 (1:1 ratio)

- group_var:

  A variable in data (single character or name) used for grouping
  observations with the same value to assign cases to train or test sets
  within a fold using
  [`rsample::group_vfold_cv()`](https://rsample.tidymodels.org/reference/group_vfold_cv.html).

## Value

List object containing cv_folds (analysis/assessment), model workflow
and seed/bag identifiers
