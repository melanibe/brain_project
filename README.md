# Analyzing brain connectivity during sleep 
## Semester project

## Build feature matrix from original matlab files 
Use the file `build_features.py` to compute and save the features matrix needed for the experiments. 
It contains the following functions:
* `prepare_X` loads and merges the original MatLab files in one single numpy array of shape [nobs, 4095, 50]. It takes a list of subject as input argument to only load data for a subset of subjects if needed. 
* `transform_X_std` performs the standard frequency band aggregation preprocessing step. It takes one matrix of size [nobs, 4095, 50] as input.
* `transform_X_one`performs the 'one' frequency band aggregation preprocessing step.

Running the file will trigger the creation and saving of:
 * the full original feature matrix of shape [nobs, 4095, 50] per subject. Each subject matrix will be saved in the `/matrices/all/` folder, the name of the array is the name of the subject.
 * the standard frequency band matrix per subject. Each subject matrix will be saved in the `/matrices/std/` folder, the name of the array is the name of the subject.
 * the one frequency band matrix per subject. Each subject matrix will be saved in the `/matrices/one/` folder, the name of the array is the name of the subject.
 * the label array per subject. Each label array will be saved in the `/matrices/y/` folder, the name of the array is the name of the subject.
              
## Graph classification network files
### Model definition

### Train utils

### Estimator wrapper


## Experiments from the report

### Cross-validation utils

### Running the experiments

