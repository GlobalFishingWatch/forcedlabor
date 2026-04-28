# Training forced labor random forest model.

Training forced labor random forest model.

## Usage

``` r
fl_train(
  cv_setup = cv_df[[1]],
  rf_spec = rf_setup$rf_spec,
  holdout = NULL,
  save_dir = NULL
)
```

## Arguments

- cv_setup:

  List containing cv_folds (analysis/assessment), model workflow and
  seed/bag identifiers. Output from ?cv_setup

- rf_spec:

  Random forest classifier specifications

- holdout:

  Optional. Test dataset, not used for model training.

- save_dir:

  Directory to save trained models otherwise skip saving when NULL

## Value

List containing: Trained random forest models Tibble with predicted
probabilities
