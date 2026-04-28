# Enforcing partial monotonicity on r

Each element of r is forced to be monotonic where, for the element, y_u
\> y_u.mean(). See Algorithm 2 in reference for more details.

## Usage

``` r
monotonize(r, y_u)
```

## Arguments

- r:

  sorted array of density ratios

- y_u:

  vector with the predictions of unlabeled

## Value

sorted array of density ratios, monotonized

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
