# Computes density kernels estimated for positive and unlabeled, and the values of those densities inferred for unlabeled predictions (sorted)

Computes density kernels estimated for positive and unlabeled, and the
values of those densities inferred for unlabeled predictions (sorted)

## Usage

``` r
kernel_unlabeled(data)
```

## Arguments

- data:

  data frame. Needs to have a .pred_1 column with predictions and a
  known_offender column with 0 for unlabeled and 1 for positive
  (offender)

## Value

a list with 3 elements: f_yp : inferred densities for positive
predictions; f_yu : inferred densities for unlabeled predictions; y_u :
vector with the predictions of unlabeled
