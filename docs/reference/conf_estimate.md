# Compute a confidence level estimate

Computing confidence level estimates for the predicted class based on a
beta distribution fitted on the random forest scores and computing the
probability mass below or above the classification threshold, if the
class is 0 or 1, respectively

## Usage

``` r
conf_estimate(predicted_df, data, threshold)
```

## Arguments

- predicted_df:

  dataframe of one row containing the predicted class and the ID of the
  vessel, that should match the IDs in data

- data:

  dataframe with all the random forest scores

- threshold:

  threshold to classify the scores into 0 or 1

## Value

confidence level estimates
