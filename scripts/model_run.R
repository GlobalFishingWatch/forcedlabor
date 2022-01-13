
# library(magrittr)

# In the VM, I need to work in the same space (have a README about it)


###################### data pre-processing ##########################

# Establish connection to BigQuery project
con <- DBI::dbConnect(drv = bigrquery::bigquery(),
                      project = "world-fishing-827", use_legacy_sql = FALSE)
# Deal with BQ data download error
# See for reference: https://github.com/r-dbi/bigrquery/issues/395
options(scipen = 20)
query_path <- "./queries/"
gfw_project <- "world-fishing-827"
### Loading offenders vessel-year dataset

ais_fl <- glue::glue(
  "SELECT
      *
    FROM
      `scratch_rocio.all_fl_ais`
"
)

fl_df <- fishwatchr::gfw_query(query = ais_fl, run_query = TRUE, con = con)

# write_csv(fl_df$data,here::here("data","offender_data.csv"))
# fl_df$data <- read_csv(here::here("data","offender_data.csv"))

### Loading all together vessel-year dataset (tidy version)

ais_tidy <- glue::glue(
  "SELECT
      *
    FROM
      `scratch_rocio.all_together_tidy`
"
)

alltogether_tidy <- fishwatchr::gfw_query(query = ais_tidy, run_query = TRUE, con = con)

gears_interest <- c("drifting_longlines", "trawlers", "squid_jigger",
                    "purse_seines", "tuna_purse_seines", "set_longlines")
# turn to factor the categorical variables that I need in the model
vars_to_factor <- c("gear", "ais_type", "foc", "source_id",
                    "known_offender")
# remove variables not needed at all
vars_remove <- c("shipname", "registry_shipname", "imo", "ais_imo", "callsign",
                 "ais_callsign",
                 "positions", "overlap_hours_multinames", "on_fishing_list_best",
                 "iuu", "number_iuu_encounters",
                 "fl_abuse_of_vulnerability", "fl_deception",
                 "fl_restriction_of_movement", "fl_isolation",
                 "fl_physical_and_sexual_violence", "fl_intimidation_and_threats",
                 "fl_retention_of_identity_documents", "fl_withholding_of_wages",
                 "fl_debt_bondage", "fl_abusive_working_and_living_conditions",
                 "fl_excessive_overtime", "fl_trafficking")

### before any filtering, we need to preprocess the data,
# particularly because we will switch some gears manually

data_preprocess <- ml_prep_data(fl_data = fl_df$data,
                                  tidy_data = alltogether_tidy$data ,
                                  gears_interest, vars_to_factor,
                                  vars_remove)

training_df <- data_preprocess$training_set
# save training_df for making summary figure (needs the fl indicators)
# readr::write_csv(training_df,here::here("data","training_df.csv"))

training_df <- training_df %>%
  dplyr::select(-all_of(vars_remove))

prediction_df <- data_preprocess$holdout_set
# readr::write_csv(prediction_df,here::here("data","holdout_df.csv"))


################## writing the recipe ##########################################

fl_rec <- recipes::recipe(known_offender ~ .,  # modeling known_offender using everything else
                 data = training_df) %>%
  # actually some are more id variables
  recipes::update_role(indID,
              new_role = "id") %>%
  # actually I don't want to use other variables in the model
  recipes::update_role(ssvid, year, flag,
              new_role = "dont_use") %>%
  # and some others will be useful for preventing data leakage
  recipes::update_role(fl_event_id, source_id, known_non_offender, possible_offender,
              year_focus, num_years, past_ais_year,
              event_ais_year, new_role = "control") %>%
  # Remove near-zero variance numeric predictors
  recipes::step_nzv(recipes::all_predictors())  %>%  # almost zero variance removal (I don't think we have those though)
  # Remove numeric predictors that have correlation greater the 75%
  recipes::step_corr(recipes::all_numeric(), threshold = 0.75)


######### specifying the model #################################################

# RF with hyperparameters to tune
rf_spec <-
  # type of model # if no tuning # rand_forest()
  parsnip::rand_forest(trees = 500,
              # We will tune these two hyperparameters
              mtry = tune(),
              min_n = tune()) %>%
  # mode
  parsnip::set_mode("classification") %>%
  # engine/package
  parsnip::set_engine("ranger", regularization.factor = tune())


########### training and testing scheme ########################################

## defining some parameter values ##
num_folds <- 10 # number of folds
num_bags <- 10 #10,20,30,50,100 # Keep this low for now for speed, but can crank up later
down_sample_ratio <- 1 # downsampling ratio
# Set common seed to use anywhere that uses random numbers
# We'll vary this to get confidence intervals
# Eventually we can crank this up (16,32,64), but keep it to 2 for now for testing
num_common_seeds <- 4
common_seed_tibble <- tibble::tibble(common_seed = seq(1:num_common_seeds)*101)

# Run all common_seeds
bag_runs <- common_seed_tibble %>%
  # For each common_seed, run all bags
  tidyr::crossing(tibble::tibble(bag = seq(num_bags)))%>%
  # Will use different random seeds when implementing recipes for each bag
  dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) %>%
  # counter
  dplyr::mutate(counter = dplyr::row_number())

## parallelization strategy
parallel_plan <- "multicore" # multisession if running from RStudio, or
# multicore if from Linux, Mac and plain R, or
# psock if multisession is not working well and you need to try something else
free_cores <- 1 # add more if you need to do many things at the same time


## CROSS VALIDATION ##
# Ensure there is no splitting across source_id across analysis and assessment
# data sets.  Need to make separate splits for each seed.
cv_splits_all <- common_seed_tibble %>%
  dplyr::mutate(cv_splits = purrr::map(common_seed,function(x){
    set.seed(x)
    rsample::group_vfold_cv(training_df,
                   group = source_id,
                   v = num_folds)
  }))

### FIRST TRAINING STAGE ###

tictoc::tic()
train_pred_proba <- ml_training(training_df = training_df, fl_rec = fl_rec,
                                     rf_spec = rf_spec, cv_splits_all = cv_splits_all, #common_seed_tibble,
                                     # num_folds,
                                     bag_runs = bag_runs, down_sample_ratio = down_sample_ratio, num_grid = 2,
                                     parallel_plan = parallel_plan, free_cores = free_cores)
tictoc::toc()


###### finding the optimal threshold and hyperparameters #########

tictoc::tic()
best_hyperparameters <- ml_hyperpar(train_pred_proba)
# write_csv(best_hyperparameters,here::here("outputs/stats", "best_hyperpar.csv"))
tictoc::toc()


####### Frankenstraining ########################################

tictoc::tic()
cv_model_res <- ml_frankenstraining(training_df = training_df,
                                    fl_rec = fl_rec,
                                    rf_spec = rf_spec,
                                    cv_splits_all = cv_splits_all,
                                    bag_runs = bag_runs,
                                    down_sample_ratio = down_sample_ratio,
                                    # rf_feature_selection_threshold = rf_feature_selection_threshold,
                                    # rf_specs_imp = rf_specs_imp,
                                    parallel_plan = parallel_plan,
                                    free_cores = free_cores,
                                    best_hyperparameters = best_hyperparameters,
                                    prediction_df = prediction_df,
                                    run_dalex = FALSE)
tictoc::toc()
