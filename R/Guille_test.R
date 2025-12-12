rm(list = ls())


# from the using-package Rmd ----------------------------------------------

#GM: Difference between both datasets?
ddir<-"./data"
ddir_raw<-"./data-raw"
load(file=file.path(ddir_raw,"fl_training.rda"))
dim(new_training_data)
load(file=file.path(ddir,"fl_training.rda"))
dim(fl_training)

#GM: this seems redundat as the known_non_offender is already a factor str(fl_training)
fl_training$known_non_offender <- as.factor(fl_training$known_non_offender)

#GM: Why this line of code?
levels(fl_training$known_non_offender) <-
  c(levels(fl_training$known_non_offender),1)


#GM: Why this is not wrapped in a function?
num_folds <- 5
num_bags <- 5
down_sample_ratio <- 1
num_common_seeds <- 3
common_seed_tibble <- tibble::tibble(common_seed =
                                       seq(1:num_common_seeds) * 101)


# Run all common_seeds
bag_runs <- common_seed_tibble |>
  tidyr::crossing(tibble::tibble(bag = seq(num_bags))) |>
  dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) |>
  dplyr::mutate(counter = dplyr::row_number())

## Cross Validation
# Ensure there is no splitting across source_id across analysis and assessment
# data sets.  Need to make separate splits for each seed.
cv_splits_all <- common_seed_tibble |>
  dplyr::mutate(cv_splits = purrr::map(common_seed, function(x) {
    set.seed(x)
    rsample::group_vfold_cv(fl_training,
                            group = source_id_number,
                            v = num_folds)
  }))
#GM: data in each split: cv_splits_all$cv_splits[[1]]$splits[[1]]$data

#test<-apply(common_seed_tibble, MARGIN = 1, FUN = function(x){
#  set.seed(x)
#  rsample::group_vfold_cv(fl_training,
#                          group = source_id_number,
#                          v = num_folds)
#},simplify = F)
#cv_splits_all$test<-test


fl_rec <- recipes::recipe(known_offender ~ .,
                          data = fl_training) |>
  recipes::update_role(indID,
                       new_role = "id") |>
  recipes::update_role(flag_region, known_non_offender,
                       new_role = "dont_use")  |>
  recipes::update_role(source_id_number, new_role = "control")  |>
  recipes::step_nzv(recipes::all_predictors())  |>
  recipes::step_corr(recipes::all_numeric(), threshold = 0.75)

rf_spec <-
  parsnip::rand_forest(trees = 500,
                       mtry = 1,
                       min_n = 15) |>
  parsnip::set_mode("classification") |>
  parsnip::set_engine("ranger", regularization.factor = 0.5)


## parallelization strategy
parallel_plan <- "multisession" # multisession if running from RStudio, or
# multicore if from Linux, Mac and plain R, or
# psock if multisession is not working well and you need to try something else
if (parallel_plan == "multisession") {
  utils::globalVariables("multisession")
}
free_cores <- 4 # add more if you need to do many things at the same time
oopts <- options(future.globals.maxSize = 10200*1024^2)  ## 15 GB
# options(future.globals.maxSize = 50 * 1024 ^ 3)

# Guille´s functions ------------------------------------------------------

# I have split ml_train_predict in three functions:
# bag_downsample: get a recipe with downsampling for each bag and corresponding seed (nested within the cv_setup)
# cv_setup: set ups the workflow and cv folds. Returns cv_workflow, cv_folds, seed and bag (for training the model in mk_train)
# ml_train: train and predict over the test set. Return each fitted model and the predicted df of scores over the test set

### First function
bag_downsample<-function(bag_runs,
                         fl_rec,
                         down_sample_ratio){
  bag_runs |>
    dplyr::mutate(
      # get a recipe with downsampling for each bag and corresponding seed
      fl_recipe = purrr::map(.data$recipe_seed, function(x) {
        fl_rec_down <- fl_rec |>
          themis::step_downsample(known_offender,
                                  under_ratio = down_sample_ratio,
                                  seed = x,
                                  skip = TRUE)
      })
    )
}

### Next would be the creation of the different CV folds
cv_setup <- function(bag_runs,
                     cv_splits_all,
                     rf_spec)
  {

  down_bags<-bag_downsample(bag_runs = bag_runs,
                            fl_rec = fl_rec,
                            down_sample_ratio = down_sample_ratio)

  out<-purrr::pmap(list(down_bags$fl_recipe,
                        down_bags$common_seed,
                        down_bags$bag), # previously future_map2, now pmap to map 3 inputs
                   function(x, y, .bag) # added .data$bag as third mapped input
                   {

                     # Ensure all bags look the same
                     set.seed(y)

                     # specifying the workflow with the model, recipe for data and how the
                     # tuning goes
                     cv_predictions_workflow <-
                       workflows::workflow() |>
                       workflows::add_model(rf_spec) |>
                       workflows::add_recipe(x)

                     # get the folds related to that common seed, train and predict
                     cv_predictions <-
                       cv_splits_all |>
                       dplyr::filter(.data$common_seed == y) |>
                       purrr::pluck('cv_splits',1) |> # unlist first (unique) element
                       dplyr::mutate(# Create analysis dataset based on CV folds
                         analysis = purrr::map(.data$splits, ~rsample::analysis(.x)),
                         # Create assessment dataset based on CV folds
                         assessment = purrr::map(.data$splits, ~rsample::assessment(.x))) |>
                       dplyr::select(-.data$splits)

                     return(
                       list(
                         workflow   = cv_predictions_workflow,
                         cv_folds = cv_predictions,
                         seed = y,
                         bag = .bag
                         )
                       )

                   })

  return(out)
}

# Next we train the models:
ml_train_new <- function(cv_setup,
                         free_cores,
                         parallel_plan,
                         save_dir){
  # Setting up the parallelization
  if (parallel_plan == "multicore") {
    future::plan(future::multicore,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
    # the garbage collector will run automatically (and asynchronously) on the
    # workers to minimize the memory footprint of the worker.
  } else if (parallel_plan == "psock") {
    cl <- parallelly::makeClusterPSOCK(parallelly::availableCores() - free_cores)
    future::plan(future::cluster, workers = cl)
  } else {
    future::plan(future::multisession,
                 workers = parallel::detectCores() - free_cores, gc = TRUE)
  }

  out <- furrr::future_pmap(
    list(
      purrr::map(cv_setup, "workflow"),
      purrr::map(cv_setup, "seed"),
      purrr::map(cv_setup, "bag"),
      purrr::map(cv_setup, "cv_folds")
    ),
    function(workflow, seed, bag, folds_tbl){

      if (!"themis" %in% loadedNamespaces())
        requireNamespace("themis", quietly = TRUE)

      out_2 <- purrr::pmap(
        list(
          folds_tbl$analysis,
          folds_tbl$assessment,
          folds_tbl$id
        ),
        function(ind_anal, ind_assess, fold_id){

          # reproducible within each outer future worker
          set.seed(seed)

          # fit the model
          tmp_model <- workflows:::fit.workflow(workflow, ind_anal)

          # save model
          file_name <- file.path(
            save_dir,
            paste0("rf_seed", seed, "_bag", bag, "_", fold_id, ".rds")
          )
          saveRDS(tmp_model, file_name)

          tmp_pred_assess <- workflows:::predict.workflow(
            object = tmp_model,
            new_data = ind_assess,
            type = "prob"
          ) |>
            dplyr::select(.pred_1) |>
            dplyr::bind_cols(ind_assess[c("indID","known_offender","known_non_offender")]) |>
            dplyr::mutate(
              holdout = 0,
              common_seed = seed,
              bag = bag,
              id = fold_id
            )

          return(list(
            model = tmp_model,
            pred_assess = tmp_pred_assess
          ))
        }
      )

      # return aggregated outputs for this workflow/seed/bag
      return(list(
        models = purrr::map(out_2, "model"),
        pred_assess = dplyr::bind_rows(purrr::map(out_2, "pred_assess"))
      ))
    },
    .options = furrr::furrr_options(seed = TRUE, packages = c("themis"))
  )

  if (parallel_plan == "psock") {
    parallel::stopCluster(cl)
  }

  list(
    models = purrr::map(out, "models"),
    train_pred_proba = dplyr::bind_rows(purrr::map(out, "pred_assess"))
  )
}

# GM: testing the functions over the first two bags
tictoc::tic()
cv_df<- cv_setup(bag_runs = bag_runs[1:2,],
                 cv_splits_all = cv_splits_all,
                 rf_spec = rf_spec)
train_test <- ml_train_new(cv_setup = cv_df,
                           free_cores = free_cores,
                           parallel_plan = parallel_plan,
                           save_dir = "./models/test")
tictoc::toc() #6.773 sec elapsed




# GM: original ml_train_predict function
# Running it for the first two bags only
source("./R/ml_train_predict.R")
tictoc::tic()
train_pred_proba2 <- ml_train_predict(
  fl_rec = fl_rec,
  rf_spec = rf_spec,
  cv_splits_all = cv_splits_all,
  bag_runs = bag_runs[1:2,],
  down_sample_ratio = down_sample_ratio,
  parallel_plan = parallel_plan,
  free_cores = free_cores,
  prediction_df = NULL,
  save_dir = "models"
)
tictoc::toc()


#GM:
#Checking .pred1 (predictions over the test set are equivalent in both methods)
all(train_pred_proba2$.pred_1 == train_test$train_pred_proba$.pred_1)

# GM:
# Function to read saved models based on cv_folds and save_dir:
ml_read_fit <- function(cv_folds,
                        save_dir){
  lapply(cv_df, function(x){
    sapply(x$cv_folds$id, function(y){
      readRDS(file=file.path(save_dir,paste0("rf_seed", x$seed, "_bag", x$bag, "_", y, ".rds")))
    },simplify = F)
  })
}

ml_fitted_new <- ml_read_fit(cv_folds=cv_df$cv_folds,save_dir = "./models/test")
ml_fitted_original <- ml_read_fit(cv_folds=cv_df$cv_folds,save_dir = "./models")

# GM: testing the models predict the same under both methods
lapply(cv_df, function(x){
  sapply(seq_along(x$cv_folds$id), function(y){
    print(paste0(x$seed,"_","bag",x$bag,"_",x$cv_folds$id[y]))

    m <- readRDS(paste0("/Users/gmartin/Documents/git/GFW/forcedlabor/models/test/rf_seed",
                        x$seed,"_","bag",x$bag,"_",x$cv_folds$id[y],".rds"))

    m_original <- readRDS(paste0("/Users/gmartin/Documents/git/GFW/forcedlabor/models/rf_seed",
                                 x$seed,"_","bag",x$bag,"_",x$cv_folds$id[y],".rds"))

    ind_assess<-x$cv_folds$assessment[[y]]

    preds_m <-
      workflows:::predict.workflow(object = m,
                                   new_data = ind_assess,
                                   type = "prob") |>
      dplyr::select(.data$.pred_1) |>
      # Add columns to assessment data
      dplyr::bind_cols(ind_assess[c("indID", "known_offender", "known_non_offender")]) |>
      dplyr::mutate(holdout = 0)

    preds_moriginal <-
      workflows:::predict.workflow(object = m_original,
                                   new_data = ind_assess,
                                   type = "prob") |>
      dplyr::select(.data$.pred_1) |>
      # Add columns to assessment data
      dplyr::bind_cols(ind_assess[c("indID", "known_offender", "known_non_offender")]) |>
      dplyr::mutate(holdout = 0)

    print(all(preds_m$.pred1 == preds_moriginal$.pred1))

  },simplify = F)

})
