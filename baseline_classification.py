import numpy as np
import os
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

""" Runs the 3 baselines gridsearch of my first classification report.
"""

############## PARAMS SETUP ###############
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)
parser.add_argument("-t", "--type", help="choose the type of freq aggregation")
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

try: #for local run
    os.chdir("/Users/melaniebernhardt/Documents/RESEARCH PROJECT/")
    cwd = os.getcwd()
    data_folder = cwd + '/DATA/'
except: #for cluster run
    cwd = os.getcwd()


############### LOGGER SETUP #############
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
LOG_FILENAME= cwd + '/logs_extended/' + '{}_classification_'.format(type_agg)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)



############ LOADING DATA #############
logger.info("The features matrice is {}".format(type_agg))
try:
    X = np.load(cwd+"/{}.npy".format(type_agg))
    Y = np.load(cwd+"/{}.npy".format(y))
    logger.info("loaded X and y")
except:
    logger.error("Not found. Should prepare X and Y first.")
n,m = np.shape(X)

Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=test_size, random_state=42)
logger.info("Shape of train features matrix is {}".format(np.shape(X_train)))


################ GRIDSEARCH SVC ###############
pipeSVC = Pipeline([ ('var', VarianceThreshold(threshold=0)), ('svm', SVC()) ])
param_grid = [ {'svm__kernel': ['rbf', 'linear'], 'svm__C':[1, 10]}]
logger.info("Beginning gridsearch svm")
grid = GridSearchCV(pipeSVC, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for SVM alone and type {}: \n".format(type_agg)+l)


################ GRIDSEARCH PCA+SVC ###############
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('svm', SVC()) ])
param_grid = [ {'pca__n_components':  [5, 10, 50, 100], 'svm__kernel': ['linear']}]
logger.info("Beginning gridsearch PCA + SVM")
grid = GridSearchCV(pipePCA, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for PCA + SVM and type {}: \n".format(type_agg)+l)


################## GRIDSEARCH KBEST + RF ################
pipeRF = Pipeline([('var', VarianceThreshold(threshold=0)),('Kbest', SelectKBest()), ('rf', RandomForestClassifier())])
param_grid = [{'Kbest__k': [300, 500, 1000], 'rf__n_estimators': [3000], 'rf__min_samples_split':[10, 30]}]
logger.info("Beginning gridsearch with variance threshold + KBest + RF")
grid = GridSearchCV(pipeRF, cv=3, n_jobs=njobs, param_grid=param_grid, scoring=['accuracy','f1'], refit=False, verbose=2)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_f1", ascending=False).to_string(index=False)
logger.info("Results for RF pipeline and type {}: \n".format(type_agg)+l)
