# Analyzing brain connectivity during sleep 
## Semester project

This repository contains the code associated to the `report.pdf` of the project. It contains all necessary files to build the numpy feature matrix from the original matlab files, to construct and train the neural network and to run the final experiments for which to results are presented in the report.

This readme presents the files and the functions they contain. Every function and its argument is documented precisely in the code.

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
 
Please run this file prior to running any other file of this project as the other file assume that the data is saved in a npy array following the folder structure described above.
              
## Graph classification network files
The `siamese_gcn` contains all the files necessary to build and train the grah classification network described in the report. It contains 3 files:
* `model.py`contains the convolutional layer definiton, the definition of the node classification network and the definition of the graph classification network. 
* `train_utils.py` defines all core functions to train the network
* `GCN_estimator.py` wraps the constructed network into an object of class BaseEstimator from sklearn, this allows us to use the same procedure with the baseline and with our network. 

### model.py
One layer class:
* Class `GraphConvLayer` defines the graph convolutional layer. The implementation of this class is taken from https://github.com/tkipf/pygcn.

Two network definition class. Each of these classes contains one init method and one forward method.
* Class `NodeGCN` defines the node graph convolution network to derive the nodes features.
* Class `GraphClassificationNet` defines the whole graph classification network. Combines the node classification, sum pooling, fully connected layer.

### train_utils.py
This file defines the training loop, a single training step and a single validation step. This is the main helper file for
the fit/predict functions in the `GCN_estimator.py` file.

* `training_step`: defines a single training step for one batch of observations. 1. Computes the Ahat matrix for each frequency band, for each observation in the batch.
2. Gets the outputs from the network with this input
3. Apply the optimizer
4. Compute the training balanced accuracy over the batch
 * `val_step`: defines a single validation step for the validation set.
        1. Computes the Ahat matrix for each frequency band
           for each observation in the batch.
        2. Gets the outputs from the network with this input
        3. Compute the validation balanced accuracy and loss over the batch

* `training_loop` function that runs the training loop and saves the plots of the training and validation loss and balanced accuracy. Assumes that standard frequency band aggregation preprocessing step has been performed for the feature matrix i.e. expects an input matrix of size `[ntrain, 5*nchannels]`. 

### GCN_estimator.py
This file defines a wrapper class for the GCN. This is necessary in order to use this network just as
it was any sklearn estimator in the (custom) cross validation.
It defines the `GCN_estimator_wrapper` class a child of the `BaseEstimator` and `ClassifierMixin` classes. It contains one init method, one fit method (i.e. training method, calls the training loop function defined in `train_utils.py`), one predict method (to predict labels on a test set after having fitted i.e. trained the estimator), one predict_proba method (to predict probabilities associated to each class on a test set after having fitted i.e. trained the estimator).

## Experiments from the report
### Cross-validation utils
The file `CV_utils.py`  defines all the customized cross validations procedure decribed in the experiments section of the report.

* Class `UpsampleStratifiedKFold` customizes the standard `StratifiedKFold` class available in sklearn in order to add the possiblity to perfrom upsampling on each training fold. It implements the procedure detailed in the report.

* Funcion `AcrossSubjectCV` runs the across-subject cross-validation experiment of the report. It splits the dataset such that one subject is one fold. Then it runs the cross-validation saving the results to a logger file. It also returns         results (a dataframe containing the results averaged over all folds), metrics (a dataframe containing the results per fold), confusion (a list of confusion matrix for each fold), conf_perc (a list of percentage confusion matrix for each fold).


### Running the experiments

