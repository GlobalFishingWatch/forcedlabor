# Compute D from MAX_SLOPE algorithm

Compute D from MAX_SLOPE algorithm

## Usage

``` r
compute_D(r, steps = 1000)
```

## Arguments

- r:

  sorted 1D array of density ratios

- steps:

  number of locations at which to compute D

## Value

alpha and D vectors

## Details

See Algorithm 2 in the reference. `D = alpha - mean(p(Yu))` where
`p(Yu)` is defined as `p(Yu) = min(alpha * r(Yu), 1)`

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
