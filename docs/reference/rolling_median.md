# Smoothing r using a rolling median

Smoothing r using a rolling median

## Usage

``` r
rolling_median(r, l_2 = 20)
```

## Arguments

- r:

  sorted array of density ratios

- l_2:

  denominator to get rolling window of length(r)/l_2 (default to 20
  based on the reference)

## Value

sorted array of density ratios, smoothed

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
