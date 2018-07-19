import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import logging
import time 
import argparse

""" Runs the gridsearches of classification report.
"""
# corrected version of https://stackoverflow.com/questions/30040597/how-to-generate-a-custom-cross-validation-generator-in-scikit-learn
# should put that in separated file with the other custom CV
class UpsampleStratifiedKFold:
    """ custom cv generator for upsampling.
    """
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        skf = StratifiedKFold(n_splits=self.n_splits)
        for train_idx, test_idx in skf.split(X,y):
            neg_ix = np.where([y[i]==0 for i in train_idx])[0]
            pos_ix = np.where([y[i]==1 for i in train_idx])[0]
            aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix), replace=True)
            train_ix = np.append(neg_ix, aug_pos_ix)
            yield train_ix, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

############## PARAMS SETUP ###############
gridsearch_scores = ['roc_auc','accuracy','f1']
best_score=False # change if want to refit to best.

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)
parser.add_argument("-t", "--type", help="choose the type of freq aggregation")
parser.add_argument("-s", "--test_size", help="choose test size in prop", type=float)
args = parser.parse_args()
type_agg = args.type

if args.test_size:
    test_size = args.test_size
else:
    test_size = 0.3

if args.njobs:
    njobs = args.njobs
else:
    njobs = 3

try: #for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project/")
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
LOG_FILENAME= cwd + '/logs/augmented/' + '{}_classification_'.format(type_agg)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)



############ LOADING DATA #############
logger.info("The feature matrix is {}".format(type_agg))
try:
    X = np.load(cwd+"/{}.npy".format(type_agg))
    Y = np.load(cwd+"/y.npy")
    logger.info("loaded X and y")
except:
    logger.error("Not found. Should prepare X and Y first.")
n,m = np.shape(X)

# distinguish only REM-nonREM
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=test_size, random_state=42, stratify=Y_main)
logger.info("Shape of train features matrix is {}".format(np.shape(X_train)))

################ GRIDSEARCH BASELINE ##############
# use the same cv and calculate the accuracy on dummy classifier
idiot = DummyClassifier(constant=0)
param_grid = [ {'strategy': ['uniform', 'constant']}]
logger.info("Beginning gridsearch svm")
grid = GridSearchCV(idiot, cv=UpsampleStratifiedKFold(3), n_jobs=njobs, param_grid=param_grid, \
                    scoring=gridsearch_scores, refit=False, verbose=2, 
                    return_train_score=False)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_roc_auc", ascending=False).to_string(index=False)
logger.info("Results for baselines and on augmented feature matrix {} are: \n".format(type_agg)+l)


################ GRIDSEARCH NORMALIZED + PCA + SVC ###############
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('pca', PCA()), ('std', StandardScaler()), ('svm', SVC()) ])
try:
    param_grid = [ {'pca__n_components':  [50, 100, 500], 'svm__kernel': ['linear','rbf']}]
    logger.info("Beginning gridsearch PCA + SVM")
    grid = GridSearchCV(pipePCA, cv=UpsampleStratifiedKFold(3), n_jobs=njobs, param_grid=param_grid, \
                     scoring=gridsearch_scores, refit=best_score, \
                     verbose=2, return_train_score=False)
    grid.fit(X_train, Y_train)
except:
    param_grid = [ {'pca__n_components':  [20, 50], 'svm__kernel': ['linear','rbf']}]
    grid = GridSearchCV(pipePCA, cv=UpsampleStratifiedKFold(3), n_jobs=njobs, param_grid=param_grid, \
                     scoring=gridsearch_scores, refit=best_score, \
                     verbose=2, return_train_score=False)
    grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_roc_auc", ascending=False).to_string(index=False)
logger.info("Results for PCA + SVM and on augmented feature matrix {} are: \n".format(type_agg)+l)


################## GRIDSEARCH KBEST + RF ################
pipeRF = Pipeline([('var', VarianceThreshold(threshold=0)), ('std', StandardScaler()), ('PerBest', SelectPercentile()), ('rf', RandomForestClassifier())])
param_grid = [{'PerBest__percentile': [10, 20, 50, 100], 'rf__n_estimators': [10000], 'rf__min_samples_split':[10, 30]}]
logger.info("Beginning gridsearch with variance threshold + KBest + RF")
grid = GridSearchCV(pipeRF, cv=UpsampleStratifiedKFold(3), n_jobs=njobs, param_grid=param_grid, scoring=gridsearch_scores,\
                     refit=best_score, verbose=2, return_train_score=False)
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_roc_auc", ascending=False).to_string(index=False)
logger.info("Results for RF pipeline and on augmented feature matrix {} are: \n".format(type_agg)+l)


################ GRIDSEARCH NORMALIZED SVC ###############
pipeSVC = Pipeline([ ('var', VarianceThreshold(threshold=0)), ('std', StandardScaler()), ('svm', SVC()) ])
param_grid = [ {'svm__kernel': ['rbf', 'linear'], 'svm__C':[1, 10]}]
logger.info("Beginning gridsearch svm")
grid = GridSearchCV(pipeSVC, n_jobs=njobs, param_grid=param_grid, \
                    scoring=gridsearch_scores, refit=best_score, verbose=2, \
                    return_train_score=False, cv=UpsampleStratifiedKFold(3))
grid.fit(X_train, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_roc_auc", ascending=False).to_string(index=False)
logger.info("Results for normalization and SVM alone on augmented feature matrix {} are: \n".format(type_agg)+l)
