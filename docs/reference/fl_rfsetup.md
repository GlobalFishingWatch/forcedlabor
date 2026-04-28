# Setting up the model recipe and random forest specification

Setting up the model recipe and random forest specification

## Usage

``` r
fl_rfsetup(
  training_data,
  y = "known_offender",
  x = NULL,
  id = "indID",
  dont_use = c("flag_region", "known_non_offender"),
  control = "source_id_number",
  corr_threshold = 0.75,
  rf_trees = 500,
  rf_mtry = 1,
  rf_min_n = 15,
  rf_reg_factor = 0.5
)
```

## Arguments

- training_data:

  A tibble with training data

- y:

  Character, name of the response variable. Defaults to
  `"known_offender"`

- x:

  Vector with names of the features to be used. Defaults to NULL and
  should be set explicitly to avoid adding features automatically

- id:

  ID variable name

- dont_use:

  Exclude a given variable (or variables) as predictors

- control:

  Control variable column

- corr_threshold:

  A value for the threshold of absolute correlation values. The step
  will try to remove the minimum number of columns so that all the
  resulting absolute correlations are less than this value. See
  [`recipes::step_corr()`](https://recipes.tidymodels.org/reference/step_corr.html)
  for further information

- rf_trees:

  Number of trees contained in the ensemble. The larger the number of
  trees, the more stable the predictions - but with a higher
  computational cost

- rf_mtry:

  Number of features that will be sampled at each split

- rf_min_n:

  Minimum node size (i.e. the number of vessel-years that a final node
  should have). Larger values create less complex trees, and may result
  in underfitting. Smaller values create more complex trees, and may
  result in overfitting

- rf_reg_factor:

  Regularization factor works by penalizing new variables by multiplying
  the splitting criterion by a factor. As a result, the tree is biased
  to keep splitting on features it already trusts, rather than
  constantly introducing fresh ones

## Value

List containing random forest recipe and specifications
