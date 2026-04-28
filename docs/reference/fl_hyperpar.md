# Get best hyperparameter combination for each common seed after ML training

Get best hyperparameter combination for each common seed after ML
training

## Usage

``` r
fl_hyperpar(data)
```

## Arguments

- data:

  data frame of train cross-validated datasets with several bags, it
  must have columns: .pred_1 : probability of being an offender; bag:
  bag ID; known_offender: 0 if not, 1 if yes; .row: row ID common_seed:
  common seed to generate bags

## Value

data frame of best hyperparameter combinations per common seed
