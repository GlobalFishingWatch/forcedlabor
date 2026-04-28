# Computes binary classification via DEDPUL

For each vessel-year, it computes a binary classification, 0 non
offender and 1 offender. It is based on the DEDPUL algorithm in the
reference.

## Usage

``` r
fl_classification(
  data,
  steps = 1000,
  plotting = FALSE,
  filepath = NULL,
  threshold = seq(0, 0.99, by = 0.01),
  eps = 0.01,
  confidence_levels = TRUE
)
```

## Arguments

- data:

  tibble with at least a common_seed column and a prediction_output
  column The prediction_output column is a list. Each element contains a
  tibble with predictions and covariates.

- steps:

  number of locations at which to compute D

- plotting:

  if TRUE, a D vs. alpha plot is generated

- filepath:

  if plotting is TRUE, a filepath of where to save the plot is needed

- threshold:

  potential thresholds to test

- eps:

  accepted difference (tolerance) between alpha and the actual
  proportion of positives for a given threshold

- confidence_levels:

  Boolean to compute confidence levels

## Value

tibble with classification and calibrated threshold used for them

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
