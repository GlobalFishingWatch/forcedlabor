# forcedlabor 1.0

This is a major release of the package, with several changes and additions. The main changes are the following: 


## New functions and documentation

- The RF recipes and variables preprocessing are now wrapped in two functions, `fl_rfsetup()` and `fl_cvsetup()`
- A new sample dataset, `fl_sample_data` was added with anonymized data
- A new vignette was added with the explanation of the workflow, showing how to go through it with the sample data
- All exported functions were renamed and are now prefixed `fl_`
- Function `ml_train_predict()` is now` fl_train()` and it works now for a single combination of seed and bag--not the whole batch of models--which facilitates parallelization when needed. 
- All parallelization options were taken out of each function and should now be set up by the user in the main environment.
