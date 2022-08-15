### installing and loading package ###

if (!require("forcedlabor")) {
  credentials::set_github_pat()
  devtools::install_github("GlobalFishingWatch/forcedlabor")
}
library(forcedlabor)


###################### data pre-processing ##########################

# Establish connection to BigQuery project
con <- DBI::dbConnect(drv = bigrquery::bigquery(),
                      project = "world-fishing-827", use_legacy_sql = FALSE)
# Deal with BQ data download error
# See for reference: https://github.com/r-dbi/bigrquery/issues/395
options(scipen = 20)
gfw_project <- "world-fishing-827"
### Loading offenders vessel-year dataset

ais_fl <- glue::glue(
  "SELECT
      *
    FROM
      `prj_forced_labor.all_fl_ais`
"
)

fl_df <- fishwatchr::gfw_query(query = ais_fl, run_query = TRUE, con = con)

### Loading all together vessel-year dataset (tidy version)

ais_tidy <- glue::glue(
  "SELECT
      *
    FROM
      `prj_forced_labor.all_together_tidy`
"
)

alltogether_tidy <- fishwatchr::gfw_query(query = ais_tidy, run_query = TRUE,
                                          con = con)

gears_interest <- c("drifting_longlines", "trawlers", "squid_jigger",
                    "purse_seines", "tuna_purse_seines", "set_longlines")
# turn to factor the categorical variables that I need in the model
vars_to_factor <- c("gear", "ais_type", "foc", "source_id",
                    "known_offender")
# remove variables not needed at all
vars_remove <- c("shipname", "registry_shipname", "imo", "ais_imo", "callsign",
                 "ais_callsign",
                 "positions", "overlap_hours_multinames",
                 "on_fishing_list_best",
                 "iuu", "number_iuu_encounters",
                 "fl_abuse_of_vulnerability", "fl_deception",
                 "fl_restriction_of_movement", "fl_isolation",
                 "fl_physical_and_sexual_violence",
                 "fl_intimidation_and_threats",
                 "fl_retention_of_identity_documents",
                 "fl_withholding_of_wages",
                 "fl_debt_bondage", "fl_abusive_working_and_living_conditions",
                 "fl_excessive_overtime", "fl_trafficking")

### before any filtering, we need to preprocess the data,
# particularly because we will switch some gears manually

data_preprocess <- ml_prep_data(fl_data = fl_df$data,
                                  tidy_data = alltogether_tidy$data,
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

fl_rec <- recipes::recipe(known_offender ~ .,
                          # modeling known_offender using everything else
                 data = training_df) %>%
  # actually some are more id variables
  recipes::update_role(indID,
              new_role = "id") %>%
  # actually I don't want to use other variables in the model
  recipes::update_role(ssvid, year, flag,
              new_role = "dont_use") %>%
  # and some others will be useful for preventing data leakage
  recipes::update_role(fl_event_id, source_id, known_non_offender,
                       possible_offender, year_focus, num_years, past_ais_year,
                       event_ais_year, new_role = "control") %>%
  # Remove near-zero variance numeric predictors
  recipes::step_nzv(recipes::all_predictors())  %>%
  # almost zero variance removal (I don't think we have those though)
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
num_bags <- 5 #10,20,30,50,100 # Keep this low for now for speed,
# but can crank up later
down_sample_ratio <- 1 # downsampling ratio
# Set common seed to use anywhere that uses random numbers
# We'll vary this to get confidence intervals
# Eventually we can crank this up (16,32,64), but keep it to 2 for now for
# testing
num_common_seeds <- 2
common_seed_tibble <- tibble::tibble(common_seed =
                                       seq(1:num_common_seeds) * 101)

# Run all common_seeds
bag_runs <- common_seed_tibble %>%
  # For each common_seed, run all bags
  tidyr::crossing(tibble::tibble(bag = seq(num_bags))) %>%
  # Will use different random seeds when implementing recipes for each bag
  dplyr::mutate(recipe_seed = dplyr::row_number() * common_seed) %>%
  # counter
  dplyr::mutate(counter = dplyr::row_number()) # %>%
  # # new feature for sensitivity
  # dplyr::mutate(bag_group = rep(seq(3), 3 * num_common_seeds))

## parallelization strategy
parallel_plan <- "multicore" # multisession if running from RStudio, or
# multicore if from Linux, Mac and plain R, or
# psock if multisession is not working well and you need to try something else
free_cores <- 1 # add more if you need to do many things at the same time


## CROSS VALIDATION ##
# Ensure there is no splitting across source_id across analysis and assessment
# data sets.  Need to make separate splits for each seed.
cv_splits_all <- common_seed_tibble %>%
  dplyr::mutate(cv_splits = purrr::map(common_seed, function(x) {
    set.seed(x)
    rsample::group_vfold_cv(training_df,
                   group = source_id,
                   v = num_folds)
  }))

### FIRST TRAINING STAGE ###

tictoc::tic()
train_pred_proba <- ml_training(training_df = training_df, fl_rec = fl_rec,
                                rf_spec = rf_spec,
                                cv_splits_all = cv_splits_all,
                                bag_runs = bag_runs,
                                down_sample_ratio = down_sample_ratio,
                                num_grid = 5, parallel_plan = parallel_plan,
                                free_cores = free_cores)
tictoc::toc()


###### finding the optimal threshold and hyperparameters #########

tictoc::tic()
best_hyperparameters <- ml_hyperpar(train_pred_proba)#ml_hyperpar_sensitivity(train_pred_proba)
# write_csv(best_hyperparameters,here::here("outputs/stats",
# "best_hyperpar.csv"))
tictoc::toc()


####### Frankenstraining ########################################

tictoc::tic()
cv_model_res <- ml_frankenstraining(training_df = training_df,
                                    fl_rec = fl_rec,
                                    rf_spec = rf_spec,
                                    cv_splits_all = cv_splits_all,
                                    bag_runs = bag_runs,
                                    down_sample_ratio = down_sample_ratio,
                                    parallel_plan = parallel_plan,
                                    free_cores = free_cores,
                                    best_hyperparameters = best_hyperparameters,
                                    prediction_df = prediction_df)
tictoc::toc()

####### Classification with dedpul ########################################

tictoc::tic()
classif_res <- ml_classification(data = cv_model_res, common_seed_tibble,
                                             steps = 1000, plotting = FALSE,
                                             filepath = NULL,
                                             threshold = seq(0, .99, by = 0.01), eps = 0.01)
  # ml_classification_sensitivity(data = cv_model_res, common_seed_tibble,
  #                                steps = 1000, plotting = FALSE,
  #                                filepath = NULL,
  #                                threshold = seq(0, .99, by = 0.01), eps = 0.01)
tictoc::toc()

bigrquery::bq_table(project = "world-fishing-827",
                    table = "classif_3seeds_5groups_5bags",
                    dataset = "scratch_rocio") %>%
  bigrquery::bq_table_upload(values = classif_res,
                             fields = bigrquery::as_bq_fields(classif_res),
                             write_disposition = "WRITE_TRUNCATE")

# # Computes recall for assessment sets and specificity for holdout non offenders
#
# perf_metrics <- ml_perf_metrics(data = classif_res,
#                                 common_seed_tibble = common_seed_tibble)

##### Prediction summary: Classification between seeds #####
# This is info we can give to external collaborators

pred_class_stats <- ml_pred_summary(data = classif_res,
                                    num_common_seeds = num_common_seeds)
pred_class_stats_composite <- pred_class_stats
bigrquery::bq_table(project = "world-fishing-827",
                    table = "pred_class_stats_composite3_25bags",
                    dataset = "scratch_rocio") %>%
  bigrquery::bq_table_upload(values = pred_class_stats,
                             fields = bigrquery::as_bq_fields(pred_class_stats),
                             write_disposition = "WRITE_TRUNCATE")


### Now comparing composite model mode with outputs from each model

pred_class_composite <- pred_class_stats %>%
  dplyr::select(indID, class_mode)

# Composite-3 with 25 bags vs. each seed model with 25 bags

composite_all_1 <- purrr::map_dfr(cv_splits_all$common_seed, function(x){
  acc <- classif_res %>%
    dplyr::filter(common_seed == x) %>%
    dplyr::select(indID, pred_class) %>%
    dplyr::right_join(by = "indID", y = pred_class_composite) %>%
    yardstick::accuracy(truth = as.factor(class_mode), estimate = as.factor(pred_class)) %>%
    dplyr::select(.estimate)
})
names(composite_all_1) <- "value"

bigrquery::bq_table(project = "world-fishing-827",
                    table = "composite_all_1_matches",
                    dataset = "scratch_rocio") %>%
  bigrquery::bq_table_upload(values = composite_all_1,
                             fields = bigrquery::as_bq_fields(composite_all_1),
                             write_disposition = "WRITE_TRUNCATE")

# Composite-3 with 25 bags vs. each seed model with 5 bags
# ???

# matrix_perm <- matrix(data = perm_seeds, nrow = 5, ncol = 5, byrow = TRUE)
# composite_all_3 <- group_comparison(matrix_perm)

x <- 1

classif_res %>%
  dplyr::filter(bag_group == x) %>%
  dplyr::group_by(.)

  dplyr::select(indID, pred_class) %>%


  data %>%
  dplyr::group_by(.data$indID) %>%
  dplyr::add_count(.data$pred_class, sort = TRUE) %>%
  dplyr::slice(1) %>% # we're dealing with ties by forcing to pick one
  dplyr::mutate(class_mode = .data$pred_class,
                class_prop = n / num_common_seeds) %>%
  dplyr::select(-c(.data$pred_class, n, .data$thres, .data$common_seed))


composite_all_1 <- purrr::map2_dfr(cv_splits_all$common_seed, 1:5, function(x,y){
  acc <- classif_res %>%
    dplyr::filter(common_seed == x) %>%
    dplyr::select(indID, pred_class) %>%
    dplyr::right_join(by = "indID", y = pred_class_composite) %>%
    yardstick::accuracy(truth = as.factor(class_mode), estimate = as.factor(pred_class)) %>%
    dplyr::select(.estimate)
})


group_comparison <- function(matrix_perm){
  dplyr::as_tibble(
    apply(X = matrix_perm, MARGIN = 1, FUN = function(x){
      classif_res %>%
        dplyr::filter(.data$common_seed %in% x == TRUE) %>%
        dplyr::group_by(.data$indID) %>%
        dplyr::add_count(.data$pred_class, sort = TRUE) %>%
        dplyr::slice(1) %>%
        dplyr::mutate(class_mode_group = .data$pred_class) %>%
        dplyr::ungroup() %>%
        dplyr::select(indID, class_mode_group) %>%
        dplyr::right_join(by = "indID", y = pred_class_composite) %>%
        yardstick::accuracy(truth = as.factor(class_mode), estimate = as.factor(class_mode_group)) %>%
        dplyr::select(.estimate) %>%
        dplyr::pull()
    })
  )
}


### Now comparing composite model mode with 5 3-composite model modes

# one random permutation to make groups

perm_seeds <- sample(x = cv_splits_all$common_seed, size = 15, replace = FALSE)

# now function to compare group modes

group_comparison <- function(matrix_perm){
  dplyr::as_tibble(
    apply(X = matrix_perm, MARGIN = 1, FUN = function(x){
      classif_res %>%
        dplyr::filter(.data$common_seed %in% x == TRUE) %>%
        dplyr::group_by(.data$indID) %>%
        dplyr::add_count(.data$pred_class, sort = TRUE) %>%
        dplyr::slice(1) %>%
        dplyr::mutate(class_mode_group = .data$pred_class) %>%
        dplyr::ungroup() %>%
        dplyr::select(indID, class_mode_group) %>%
        dplyr::right_join(by = "indID", y = pred_class_composite) %>%
        yardstick::accuracy(truth = as.factor(class_mode), estimate = as.factor(class_mode_group)) %>%
        dplyr::select(.estimate) %>%
        dplyr::pull()
    })
  )
}

# now making 5 groups of three

matrix_perm <- matrix(data = perm_seeds, nrow = 5, ncol = 3, byrow = TRUE)
composite_all_3 <- group_comparison(matrix_perm)
bigrquery::bq_table(project = "world-fishing-827",
                    table = "composite_all_3_matches",
                    dataset = "scratch_rocio") %>%
  bigrquery::bq_table_upload(values = composite_all_3,
                             fields = bigrquery::as_bq_fields(composite_all_3),
                             write_disposition = "WRITE_TRUNCATE")


# now making 3 groups of five


matrix_perm <- matrix(data = perm_seeds, nrow = 3, ncol = 5, byrow = TRUE)
composite_all_5 <- group_comparison(matrix_perm)

bigrquery::bq_table(project = "world-fishing-827",
                    table = "composite_all_5_matches",
                    dataset = "scratch_rocio") %>%
  bigrquery::bq_table_upload(values = composite_all_5,
                             fields = bigrquery::as_bq_fields(composite_all_5),
                             write_disposition = "WRITE_TRUNCATE")


# stats and plots

std <- function(x, na.rm = FALSE) {
  if (na.rm) x <- na.omit(x)
  sqrt(var(x) / length(x))
}

stats_plots <- data.frame(avg = rep(NA,3), avg_minus_std = rep(NA,3), avg_plus_std = rep(NA,3))

test <- "composite_all_1_matches"
comp <- glue::glue(
  "SELECT
      *
    FROM
      `scratch_rocio.{test}`
"
)
comp_1_df <- fishwatchr::gfw_query(query = comp, run_query = TRUE, con = con)$data
stats_plots$avg[1] <- mean(comp_1_df$value)
stats_plots$avg_minus_std[1] <- stats_plots$avg[1] - std(comp_1_df$value)
stats_plots$avg_plus_std[1] <- stats_plots$avg[1] + std(comp_1_df$value)

test <- "composite_all_3_matches"
comp <- glue::glue(
  "SELECT
      *
    FROM
      `scratch_rocio.{test}`
"
)
comp_1_df <- fishwatchr::gfw_query(query = comp, run_query = TRUE, con = con)$data
stats_plots$avg[2] <- mean(comp_1_df$value)
stats_plots$avg_minus_std[2] <- stats_plots$avg[2] - std(comp_1_df$value)
stats_plots$avg_plus_std[2] <- stats_plots$avg[2] + std(comp_1_df$value)

test <- "composite_all_5_matches"
comp <- glue::glue(
  "SELECT
      *
    FROM
      `scratch_rocio.{test}`
"
)
comp_1_df <- fishwatchr::gfw_query(query = comp, run_query = TRUE, con = con)$data
stats_plots$avg[3] <- mean(comp_1_df$value)
stats_plots$avg_minus_std[3] <- stats_plots$avg[3] - std(comp_1_df$value)
stats_plots$avg_plus_std[3] <- stats_plots$avg[3] + std(comp_1_df$value)

stats_plots$seeds <- c(1,3,5)
stats_plots <- stats_plots %>%
  mutate(avg_1 = 1 - avg,
         avg_1_minus_std = avg_1 - (avg_plus_std - avg),
         avg_1_plus_std = avg_1 - (avg_minus_std - avg))


ggplot(data = stats_plots, aes(x = seeds, y = avg)) +
  geom_point() +
  geom_errorbar(aes(ymin = avg_minus_std, ymax = avg_plus_std), width = 0.2) +
  scale_x_continuous(breaks = c(1,3,5)) +
  theme_bw()


ggplot(data = stats_plots, aes(x = seeds, y = avg_1)) +
  geom_point() +
  geom_errorbar(aes(ymin = avg_1_minus_std, ymax = avg_1_plus_std), width = 0.2) +
  scale_x_continuous(breaks = c(1,3,5)) +
  ylim(c(0, 0.04)) +
  theme_bw()

# if we want to save everything together
pred_stats_set <- training_df %>%
  rbind.data.frame(prediction_df) %>%
  dplyr::right_join(pred_class_stats, by = c("indID", "known_offender",
                                             "possible_offender",
                                             "known_non_offender",
                                             "event_ais_year"))

bigrquery::bq_table(project = "world-fishing-827",
         table = "pred_stats_per_vessel_year_dev_2021",
         dataset = "prj_forced_labor") %>%
  bigrquery::bq_table_upload(values = pred_stats_set,
                  fields = bigrquery::as_bq_fields(pred_stats_set),
                  write_disposition = "WRITE_TRUNCATE")


# reproducibility test

# doing this should return a similar outcome to the commented result

model_out <- glue::glue(
  "SELECT
      class_mode,
      class_prop
    FROM
      `prj_forced_labor.pred_stats_per_vessel_year_dev`
"
)

pred_df <- fishwatchr::gfw_query(query = model_out, run_query = TRUE,
                               con = con)$data
table(pred_df)

# or
table(pred_stats_set$class_mode, pred_stats_set$class_prop)

#               class_prop
# class_mode   0.5  0.75     1
          # 0  402  2984 77733
          # 1  2110  3447 27977
