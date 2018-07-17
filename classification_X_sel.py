import numpy as np
import os
from build_features import prepare_X, prepare_X_std_freq, prepare_X_one_freq, prepare_X_ten_freq
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import logging
import time 
import argparse
import matplotlib.pyplot as plt 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

""" This is the file containing all (or nearly) the GridSearch I performed on X_sel_aug (i.e. the "best" feature matrix I found).
Do not run the file as is it would take ages, comment out the gridsearch you don't need.
"""


############# SETUP ###############
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)
parser.add_argument("-t", "--type", help="choose the type feature matrix")
parser.add_argument("-y", "--y", help="choose y")
parser.add_argument("-s", "--test_size", help="choose test size in prop", type=float)
args = parser.parse_args()
type_agg = args.type

if args.test_size:
    test_size = args.test_size
else:
    test_size = 0.1

if args.njobs:
    njobs = args.njobs
else:
    njobs = 3

if args.y:
    y = args.y
else:
    y = 'y'

try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/RESEARCH PROJECT/")
    cwd = os.getcwd()
    data_folder = cwd + '/DATA/'
except: 
    cwd = os.getcwd()


# ---------- LOGGER SETUP -------- #
# create logger
global logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger('my_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
time.time()
t = time.strftime('%d%b%y_%H%M%S')
LOG_FILENAME= cwd + '/logs_extended/' + '{}_classification_indepth_'.format(type_agg)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

logger.info("The feature matrix is {}".format(type_agg))


############ LOADING DATA #############
try:
    X = np.load(cwd+"/{}.npy".format(type_agg))
    Y = np.load(cwd+"/{}.npy".format(y))
    logger.info("loaded X and y")
except:
    logger.error("Not found Have to prepare X and Y first")
n,m = np.shape(X)

Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=test_size, random_state=42)
logger.info("Shape of train features matrix is {}".format(np.shape(X_train)))


######################################### RANDOM FOREST ########################################

#### Gridsearch RF
pipeRF = Pipeline([('var', VarianceThreshold(threshold=0)),('Kbest', SelectKBest()), ('rf', RandomForestClassifier())])
param_grid = [{'Kbest__k': [100, 300, 500, 1000], 'rf__n_estimators': [3000], 'rf__min_samples_split':[10, 30, 50]}]
logger.info("Beginning gridsearch with variance threshold + KBest + RF")
grid = GridSearchCV(pipeRF, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for RF pipeline and type {}: \n".format(type_agg)+l)



######################################### SVC ########################################


#### Gridsearch SVC
pipeSVC = Pipeline([ ('var', VarianceThreshold(threshold=0)), ('svm', SVC()) ])
param_grid = [ {'svm__kernel': ['rbf', 'linear'], 'svm__C':[30, 40, 50]}]
logger.info("Beginning gridsearch svm")
grid = GridSearchCV(pipeSVC, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for SVM alone and type {}: \n".format(type_agg)+l)

#### Gridsearch PCA + SVC
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('svm', SVC()) ])
param_grid = [ {'pca__n_components': [25, 50, 100], 'svm__kernel': ['linear','rbf'], 'svm__C': [40, 50, 60, 100]}]
logger.info("Beginning gridsearch PCA + SVM")
grid = GridSearchCV(pipePCA, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for PCA + SVM and type {}: \n".format(type_agg)+l)

#### Gridsearch SelectFromModel + SVC
pipeSVC = Pipeline([ ('var', VarianceThreshold(threshold=0)),('select', SelectFromModel(LinearSVC(C=40))), ('svm', LinearSVC(C=40)) ])
param_grid = [ {'select__threshold':[None, 'median', 'mean']}]
logger.info("Beginning gridsearch SelectFromModel + SVM")
grid = GridSearchCV(pipeSVC, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for SelectFromModel + SVM: \n"+l)

#### Gridsearch SelectKBest + LinearSVC
pipeSVC = Pipeline([ ('var', VarianceThreshold(threshold=0)),('select', SelectKBest()), ('svm', LinearSVC(C=40)) ])
param_grid = [ {'select__k':[1000, 5000, 10000]}]
logger.info("Beginning gridsearch SelectK + SVM")
grid = GridSearchCV(pipeSVC, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for SelectK + SVM: \n"+l)




######################################### LDA ########################################
#### Gridsearch LDA
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('lda', LinearDiscriminantAnalysis()) ])
param_grid = [ {'pca__n_components': [20, 50, 100, 200]}]
logger.info("Beginning gridsearch PCA + LDA")
grid = GridSearchCV(pipePCA, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for PCA + LDA: \n"+l)

#### Gridsearch Kbest + LDA
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('select', SelectKBest()), ('lda', LinearDiscriminantAnalysis()) ])
param_grid = [ {'select__k': [100, 200, 1000]}]
logger.info("Beginning gridsearch PCA + LDA")
grid = GridSearchCV(pipePCA, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for SELECT + LDA: \n"+l)

#### Gridsearch KBest + PCA + LDA
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('select', SelectKBest()), ('pca', PCA()), ('lda', LinearDiscriminantAnalysis()) ])
param_grid = [ {'select__k': [2000, 5000], 'pca__n_components':[20,50,100]}]
logger.info("Beginning gridsearch SELECT + PCA + LDA")
grid = GridSearchCV(pipePCA, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for PCA + LDA: \n"+l)


######################################### MLP ########################################
#### Gridsearch small MLP
pipeMLP = Pipeline([('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('mlp', MLPClassifier(early_stopping=True))])
param_grid = [{'pca__n_components': [20, 30, 50, 100, 1000], 'mlp__hidden_layer_sizes': [(20,10),(10,5), (50,25)]}]
logger.info("Beginning gridsearch with variance threshold + PCA + MLP")
grid = GridSearchCV(pipeMLP, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for RF pipeline and type {}: \n".format(type_agg)+l)


#### Gridsearch medium MLP
pipeMLP = Pipeline([('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('mlp', MLPClassifier(early_stopping=True))])
param_grid = [{'pca__n_components': [50, 100], 'mlp__hidden_layer_sizes': [(100,50), (250,100,50)]}]
logger.info("Beginning gridsearch with variance threshold + PCA + MLP")
grid = GridSearchCV(pipeMLP, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for MLP pipeline and type {}: \n".format(type_agg)+l)


#### Grid search big MLP 
pipeMLP = Pipeline([('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('mlp', MLPClassifier(early_stopping=True))])
param_grid = [{'pca__n_components': [500, 1000, 2000], 'mlp__hidden_layer_sizes': [(5000, 2500, 1000, 500, 200, 100, 50), (2000, 1000, 400, 200, 100, 50)]}]
logger.info("Beginning gridsearch with variance threshold + PCA + MLP")
grid = GridSearchCV(pipeMLP, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for MLP pipeline \n"+l)

pipeMLP = Pipeline([('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('mlp', MLPClassifier(early_stopping=True))])
param_grid = [{'pca__n_components': [50,100,500], 'mlp__hidden_layer_sizes': [(2000, 1000, 400, 200, 100, 50), (1000, 500, 200, 50), (1000, 500, 250, 100, 50, 20), (500, 250, 125, 50, 25)]}]
logger.info("Beginning gridsearch with variance threshold + PCA + MLP")
grid = GridSearchCV(pipeMLP, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for MLP pipeline \n"+l)


