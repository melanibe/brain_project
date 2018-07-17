import numpy as np
import os
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
import logging
import time 
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

""" Performs classification after K-means. Just preliminary test.
"""

############ PARAMS SETUP ##################
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="choose the feature matrix")
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
#    data_folder = "/cluster/scratch/melanibe/"+'/DATA/'

phases = {"REM_phasic":1,"REM_tonic":2,"S2_Kcomplex":3,"S2_plain":4,"S2_spindle":5,"SWS_deep":6}



########## LOGGER SETUP ##########
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
LOG_FILENAME= cwd + '/logs/' + '{}_Kmeans_'.format(type_agg)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)


logger.info("The type is {}".format(type_agg))


############ LOADING DATA #############
X = np.load(cwd+"/{}.npy".format(type_agg))
Y = np.load(cwd+"/{}.npy".format(y))

n,m = np.shape(X)
# X_train, X_test, Y_train_full, Y_test_full = train_test_split(X, Y, test_size=test_size, random_state=42)
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=test_size, random_state=42)
logger.info("Shape of train features matrix is {}".format(np.shape(X_train)))



############ 6 K-means and SVC ##############
logger.info("Beginning 6 K-means")
kmean = MiniBatchKMeans(n_clusters=6)
clusters = kmean.fit_predict(X_train).reshape(-1,1)
clf = SVC()
param_grid = [{'C':[1, 10, 30]}]
logger.info("Beginning gridsearch 6 K-means and svm")
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 6 K-means and SVM and type {}: \n".format(type_agg)+l)

############ 6 K-means and MLP ##############
logger.info("Beginning gridsearch 6 K-means and MLP")
clf = MLPClassifier(early_stopping=True)
param_grid = [{'hidden_layer_sizes':[(2000, 1000, 400, 200, 100, 50),(1000, 500, 200, 50), (500,250,125,75) ,(100,50,25)]}]
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 6 K-means and SVM and type {}: \n".format(type_agg)+l)

""" SAME WITH OTHER NUMBER OF CLUSTER ... lol une boucle aurait été peut etre utile. 
logger.info("Beginning 3 K-means")
kmean = MiniBatchKMeans(n_clusters=3)
clusters = kmean.fit_predict(X_train).reshape(-1,1)
clf = SVC()
param_grid = [{'C':[1, 10, 30]}]
logger.info("Beginning gridsearch 3 K-means and svm")
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 4 K-means and SVM and type {}: \n".format(type_agg)+l)


logger.info("Beginning 20 K-means")
kmean = MiniBatchKMeans(n_clusters=20)
clusters = kmean.fit_predict(X_train).reshape(-1,1)
clf = SVC()
param_grid = [{'C':[10, 30]}]
logger.info("Beginning gridsearch 20 K-means and svm")
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 20 K-means and SVM and type {}: \n".format(type_agg)+l)


logger.info("Beginning gridsearch 20 K-means and MLP")
clf = MLPClassifier(early_stopping=True)
param_grid = [{'hidden_layer_sizes':[(2000, 1000, 400, 200, 100, 50),(1000, 500, 200, 50), (500,250,125,75) ,(100,50,25)]}]
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 20 K-means and SVM and type {}: \n".format(type_agg)+l)


logger.info("Beginning 50 K-means")
kmean = MiniBatchKMeans(n_clusters=50)
clusters = kmean.fit_predict(X_train).reshape(-1,1)
clf = SVC()
param_grid = [{'C':[10, 30]}]
logger.info("Beginning gridsearch 50 K-means and svm")
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 50 K-means and SVM and type {}: \n".format(type_agg)+l)


logger.info("Beginning gridsearch 50 K-means and MLP")
clf = MLPClassifier(early_stopping=True)
param_grid = [{'hidden_layer_sizes':[(2000, 1000, 400, 200, 100, 50),(1000, 500, 200, 50), (500,250,125,75) ,(100,50,25)]}]
grid = GridSearchCV(clf, cv=5, n_jobs=njobs, param_grid=param_grid, scoring=['f1','accuracy'], refit=False, verbose=2)
grid.fit(clusters, Y_train)
logger.info("Gridsearch is done")
results =  pd.DataFrame.from_dict(grid.cv_results_)
var_names = [v for v in results.columns.values if (("mean_test_" in v) or ("std_test_" in v))] + [v for v in results.columns.values if ("param_" in v)]
l = results[var_names].sort_values("mean_test_accuracy", ascending=False).to_string(index=False)
logger.info("Results for 50 K-means and SVM and type {}: \n".format(type_agg)+l)
"""