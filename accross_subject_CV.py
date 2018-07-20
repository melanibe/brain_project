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
args = parser.parse_args()
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
LOG_FILENAME= cwd + '/logs/' + 'acrossCV_'+ t +'.log'
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
        n = len(X_train)
        assert(len(X_train)+len(X_test)==10081)
        # perform upsampling only on training fold
        neg_ix = np.where([Y_train[i]==0 for i in range(n)])[0]
        pos_ix = np.where([Y_train[i]==1 for i in range(n)])[0]
        assert(len(pos_ix)+len(neg_ix)==n)
        aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix), replace=True)
        X_train = np.reshape(np.append([X_train[i] for i in neg_ix], [X_train[i] for i in aug_pos_ix]),(len(neg_ix)+len(aug_pos_ix),-1))
        Y_train = np.append([0 for i in neg_ix], [1 for i in aug_pos_ix])
        print(np.shape(X_train))
        print("Fit the estimator")
        estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, estimator.predict(X_test)))
        print(roc_auc[-1], accuracyCV[-1])
    metrics = pd.DataFrame()
    metrics['auc'] = roc_auc
    metrics['accuracy'] = accuracyCV
    metrics.index = subject_list
    results = pd.DataFrame()
    results['mean'] = [np.mean(roc_auc), np.mean(accuracyCV)]
    results['std'] = [np.std(roc_auc), np.std(accuracyCV)]
    results['min'] = [np.min(roc_auc), np.min(accuracyCV)]
    results['max'] = [np.max(roc_auc), np.max(accuracyCV)]
    results.index=['roc_auc', 'accuracy']
    return(results, metrics)


estimator_graph = Pipeline([('var', VarianceThreshold(threshold=0)), \
                            ('std', StandardScaler()), \
                            ('PerBest', SelectPercentile(percentile=50)), \
                            ('rf', RandomForestClassifier(n_estimators=10, min_samples_split=30, n_jobs=njobs))])
        

matrix_graph = np.load(cwd+'/graph_features/std_tresh0.05.npy')
Y = np.load(cwd+'/y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]


tmp = AcrossSubjectCV(estimator_graph, matrix_graph, Y_main)
logger.info("Global results for best graph feature estimator from accross subject CV are: \n"+tmp[0].to_string())
print(tmp[1])
logger.info("Results per subject for best graph feature estimator from accross subject CV are: \n"+tmp[1].to_string())


matrix_full = np.load(cwd+'/X_sel.npy')
estimator_full = Pipeline([('var', VarianceThreshold(threshold=0)), \
                            ('std', StandardScaler()), \
                            ('PerBest', SelectPercentile(percentile=10)), \
                            ('rf', RandomForestClassifier(n_estimators=10, min_samples_split=30, n_jobs=njobs))])

tmp = AcrossSubjectCV(estimator_full, matrix_full, Y_main)
logger.info("Global results for best graph feature estimator from accross subject CV are: \n"+tmp[0].to_string())
logger.info("Results per subject for best graph feature estimator from accross subject CV are: \n"+tmp[1].to_string())