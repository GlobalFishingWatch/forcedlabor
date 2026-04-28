# Computing the proportion of positives within the unlabeled (populationwise).

Computes alpha star, or the upper bound of alpha, the proportion of
positives within the unlabeled (populationwise).

## Usage

``` r
dedpul_estimation(data, steps = 1000, plotting = FALSE, filename = NULL)
```

## Arguments

- data:

  data frame. Needs to have a .pred_1 column with predictions and a
  known_offender column with 0 for unlabeled and 1 for positive
  (offender)

- steps:

  number of locations at which to compute D

- plotting:

  if TRUE, a D vs. alpha plot is created

- filename:

  if plotting is TRUE, a filename with path is required

## Value

estimated alpha value

## Details

We first get density kernels estimated for positive and unlabeled and
the values of those densities inferred for unlabeled predictions
(sorted). Then we compute the sorted array of density ratios. We finally
use it to compute alpha star. The calculations are based in the
algorithm described in the reference.

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
