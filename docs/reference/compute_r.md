# Computes density ratios array r

Compute r using Algorithms 1 and 2 in reference.

## Usage

``` r
compute_r(f_y)
```

## Arguments

- f_y:

  list with f_yp, f_yu and y_u as elements (see details)

## Value

sorted array of density ratios, monotonized and smoothed

## Details

In f_y, f_yp is the vector of inferred densities for positive
predictions, f_yu is the vector of inferred densities for unlabeled
predictions, and y_u is the vector with the predictions of unlabeled.

## References

D. Ivanov, "DEDPUL: Difference-of-Estimated-Densities-based
Positive-Unlabeled Learning," 2020 19th IEEE International Conference on
Machine Learning and Applications (ICMLA), 2020, pp. 782-790, doi:
10.1109/ICMLA51294.2020.00128.
