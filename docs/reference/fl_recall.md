# Computes recall for assessment sets

Computes recall, for assessment sets (in model versions that did not use
them for training)

## Usage

``` r
fl_recall(data)
```

## Arguments

- data:

  tibble with at least a prediction output column (`pred_class`) and a
  `known_offender` column (whether the vessel was identified as an
  offender by reports).

## Value

recall value
