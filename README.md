forcedlabor: forced labor model package
================

Updated: 2022-02-01

# Overview

`forcedlabor` is an R package containing the codes to identify forced
labor on fishing vessels via a machine learning algorithm trained on a
combination of known forced labor cases and AIS data predictors.

# Installation

This R package is hosted on a private GitHub repository and can be
installed using the `devtools` package. Run the following code to check
if you already have `devtools` and install it if not.

``` r
# Check/install devtools
if (!require("devtools"))
  install.packages("devtools")
```

To connect to GitHub, under the hood `devtools` calls the GitHub API.
This means that you’ll need to have a personal access token (PAT). Get a
PAT [here](https://github.com/settings/tokens) and make sure the “repo”
scope is included. Save your PAT as an R environment variable (variables
that should be available to R processes) by running
`usethis::edit_r_environ()`, adding `GITHUB_PAT = "your PAT"`, and
saving the updated `.Renviron` file. You might need to restart the R
session.

Finally, after saving your GitHub PAT to your R environment, install the
`forcedlabor` package using `devtools`.

``` r
# Install fishwatchr
devtools::install_github("GlobalFishingWatch/forcedlabor")
```

# Using the package

An example of how to use it is in
[`/scripts/model_run.R`](https://github.com/GlobalFishingWatch/forcedlabor/blob/main/scripts/model_run.R)

# Structure of the repository

-   **R:** R functions of the package
-   **data:** Data files. For now, it contains a couple of output data
    objects.
-   docs: Additional documentation (empty for now)
-   **man:** Standard documentation files for R functions
-   models: Trained models, model summaries, etc. (empty for now)
-   notebooks: Jupyter notebooks and R markdown (empty for now)
-   outputs: Models results, static reports, etc.
    -   figures: Figures (empty for now)
-   queries: SQL files (empty for now)
-   **renv:** Contains files to activate the R environment with specific
    settings
-   **scripts:** Regular python and R files. Contains the script to run
    the model.
-   **.Rbuildignore** Contains names of files that should be ignored to
    build the R package
-   **.Rprofile** Used in the R session to get the R environment of this
    package
-   **.gitignore** Files to ignore by version control.
-   **.pre-commit-config.yaml** Style linting using \`pre-commit
    library.
-   **DESCRIPTION** It stores the metadata of the package
-   **NAMESPACE** It contains information of important names in the
    package to provide space for them and make the package self
    contained
-   **README.md** Top-level README on how to use this repo
-   **README.Rmd** Rmarkdown file that generates README.md
-   **forcedlabor.Rproj** Rproj file
-   **renv.lock** It contains the information on the packages used in
    the project.
