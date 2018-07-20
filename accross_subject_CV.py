import numpy as np
import pandas as pd
import os
from build_features import index_subj
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import logging
import time 
import argparse

cwd = os.getcwd()
subject_list = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)

if args.njobs:
    njobs = args.njobs
else:
    njobs = 3


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

###### ACROSS CV #####
def AcrossSubjectCV(estimator, X, Y):
    """ Implements custom CV to calculate the across subject accuracy.
    10-fold CV since we have 10 subjects.
    """
    roc_auc=[]
    accuracyCV = []
    for s in subject_list:
        idx, subj_idx = index_subj(s)
        print("Preparing fold only {}".format(s))
        X_test, Y_test = [X[i] for i in subj_idx], [Y[i] for i in subj_idx]
        print("Preparing fold except {}".format(s))
        X_train, Y_train = [X[i] for i in idx], [Y[i] for i in idx]
        print("Fit the estimator")
        assert(len(X_train)+len(X_test)==10081)
        estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, estimator.predict(X_test)))
        print(roc_auc[-1], accuracyCV[-1])
    results = pd.DataFrame()
    results['mean'] = [np.mean(roc_auc), np.mean(accuracyCV)]
    results['std'] = [np.std(roc_auc), np.std(accuracyCV)]
    results['min'] = [np.min(roc_auc), np.min(accuracyCV)]
    results['max'] = [np.max(roc_auc), np.max(accuracyCV)]
    results.index=['roc_auc', 'accuracy']
    return(results)


estimator_graph = Pipeline([('var', VarianceThreshold(threshold=0)), \
                            ('std', StandardScaler()), \
                            ('PerBest', SelectPercentile(percentile=50)), \
                            ('rf', RandomForestClassifier(n_estimators=10000, min_samples_split=30, n_jobs=njobs))])
        

matrix_graph = np.load(cwd+'/graph_features/std_tresh0.05.npy')
Y = np.load(cwd+'/y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]


print(AcrossSubjectCV(estimator_graph, matrix_graph, Y_main))
