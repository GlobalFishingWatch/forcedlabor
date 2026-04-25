forcedlabor: Global Fishing Watch’s forced labor risk machine learning
algorithm
================

Updated: 2026-04-24

# Overview

`forcedlabor` is an R package develop for training a positive–unlabeled
(PU) model to detect fishing vessels exhibiting behaviors or
characteristics consistent with forced-labor offenders.

The model is trained on a combination of confirmed cases of forced
labor, an unlabeled set of vessels for which there is no information
about the occurrence of forced labor (hence the **positive-unlabelled**
denomination), and the features of the model derive from Automatic
Identification System (AIS) data processed by Global Fishing Watch that
describes vessel characteristics and activity.

# Installation

You can install the most recent version of `forcedlabor` using:

``` r

# Check/install remotes
if (!require("remotes"))
  install.packages("remotes")

# Install the forcedlabor package

remotes::install_github("GlobalFishingWatch/forcedlabor",
                        dependencies = TRUE)
```

# Using the package

Check the package vignette for function descriptions and an example
workflow.
