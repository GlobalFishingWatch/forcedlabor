# Computes threshold for offender classification

Computes threshold for offender classification based on alpha (from the
dedpul_estimation function—see reference): which threshold would achieve
an alpha or proportion of positives within the unlabeled (more or less)
equal to alpha?

## Usage

``` r
calibrated_threshold(
  data,
  steps = 1000,
  plotting = FALSE,
  filename = NULL,
  threshold = seq(0, 0.99, by = 0.01),
  eps = 0.01
)
```

## Arguments

- data:

  data frame. Needs to have a .pred_1 column with predictions and a
  known_offender column with 0 for unlabeled and 1 for positive
  (offender)

- steps:

  number of locations at which to compute D

- plotting:

  if TRUE, a D vs. alpha plot is generated

- filename:

  if plotting is TRUE, a filename with path is required

- threshold:

  potential thresholds to test

- eps:

  accepted difference (tolerance) between alpha and the actual
  proportion of positives for a given threshold

## Value

a threshold to use

## Details

For more details on the algorithm, please see the reference.

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
