# Computes recall for assessment sets and specificity for holdout non offenders

Two performance metrics are computed: recall, for assessment sets (in
model versions that did not use them for training), and specificity for
holdout non offenders (in the same year of the
certification/inspection - if done at the end of the year)

## Usage

``` r
fl_perf_metrics(data)
```

## Arguments

- data:

  tibble with at least a prediction_output column (`pred_class`), a
  `holdout` column (whether if the observation was used in the model or
  held out), a `known_offender` column (whether the vessel was
  identified as an offender by reports), and a `known_non_offender`
  column (whether the vessel was identified as non offender by
  inspections).

## Value

tibble with recall and specificity per seed
