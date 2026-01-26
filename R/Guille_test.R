# rm(list = ls())
#
#
# # from the using-package Rmd ----------------------------------------------
#
# #GM: Difference between both datasets?
# ddir<-"./data"
# ddir_raw<-"./data-raw"
# load(file=file.path(ddir_raw,"fl_training.rda"))
# dim(new_training_data)
# load(file=file.path(ddir,"fl_training.rda"))
# dim(fl_training)

# devtools::load_all()
data("fl_training")

#GM: this seems redundant as the known_non_offender is already a factor str(fl_training)
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
# cv_setup: set ups the workflow and cv folds. Returns cv_workflow, cv_folds, seed and bag (for training the model in ml_train)
# ml_train: train and predict over the test set. Return each fitted model and the predicted df of scores over the test set
source("./R/dev_cv_setup.R")
source("./R/dev_bag_downsample.R")
source("./R/dev_ml_train.R")

# GM: testing the functions over the first 5 bags (3 seeds)
tictoc::tic()
cv_df<- dev_cv_setup(bag_runs = bag_runs,
                 cv_splits_all = cv_splits_all,
                 fl_rec = fl_rec,
                 rf_spec = rf_spec,
                 down_sample_ratio = down_sample_ratio)
train_test <- dev_ml_train(cv_setup = cv_df,
                           free_cores = free_cores,
                           parallel_plan = parallel_plan,
                           save_dir = "./models/test")
tictoc::toc() #6.773 sec elapsed # 250 sec for 5 bags




# GM: original ml_train_predict function
# Running it for the first two bags only
source("./R/ml_train_predict.R")
tictoc::tic()
train_pred_proba2 <- ml_train_predict(
  fl_rec = fl_rec,
  rf_spec = rf_spec,
  cv_splits_all = cv_splits_all,
  bag_runs = bag_runs,
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

    m <- readRDS(paste0("./models/test/rf_seed",
                        x$seed,"_","bag",x$bag,"_",x$cv_folds$id[y],".rds"))

    m_original <- readRDS(paste0("./models/rf_seed",
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
