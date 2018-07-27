import numpy as np
import pandas as pd 
import os
import logging
import time 
import argparse

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from siamese_gcn.GCN_estimator import GCN_estimator_wrapper
from CV_utils import WithinOneSubjectCV, AcrossSubjectCV

""" Runs the validation runs of classification report. For now only to use with std, other matrices not built.
"""

############## PARAMS SETUP ###############
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)
parser.add_argument("-t", "--type", help="choose the matrix")
args = parser.parse_args()

if args.type:
    mat = args.type
else:
    mat = 'std'

if args.njobs:
    njobs = args.njobs
else:
    njobs = 3

try: #for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project/")
    cwd = os.getcwd()
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
LOG_FILENAME= cwd + '/validation_logs/' + '{}_classification_'.format(mat)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# for GCN saving - shoudl put an option saving = False
checkpoint_dir = cwd + '/siamese_gcn/runs/'
checkpoint_file = checkpoint_dir + t

############### CLASSIFIERS TO EVALUATE ############
uniform = DummyClassifier(strategy='uniform')
constant = DummyClassifier(constant=0, strategy='constant')
pipePCA_SVM = Pipeline([('var', VarianceThreshold(threshold=0)),\
                            ('pca', PCA(n_components=500)),\
                            ('std', StandardScaler()), \
                            ('svm', SVC(kernel='linear', probability=True))])
pipeKBest_RF = Pipeline([('var', VarianceThreshold(threshold=0)), \
                        ('std', StandardScaler()), \
                        ('PerBest', SelectPercentile(percentile=50)),\
                         ('rf', RandomForestClassifier(n_estimators=10000, min_samples_split=30))])
GCN_estimator = GCN_estimator_wrapper(checkpoint_file, logger, 32, 64, 128, n_epochs=8, reset=True)
logger.info("GCN params 32-64-128-8 epochs")



############ WITHIN ONE SUBJECT CV - 5-FOLD FOR 4 SUBJECTS #############
reliable_subj = ['S12', 'S10', 'S04', 'S05']
estimators = [uniform, constant,  pipePCA_SVM, pipeKBest_RF, GCN_estimator]
names = ['uniform', 'constant',  'PCA and SVM', 'KBest and RF', 'GCN']

for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], subject=[subject], k=5, mat=mat)
        logger.info("Results for subject {} for estimator {}".format(subject, names[i]))
        logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
        logger.info("Results by fold: \n"+metrics.to_string())
        for k in range(len(confusion)):
            logger.info("Confusion matrices across folds are: \n"+pd.DataFrame(confusion[k]).to_string())
        logger.info("Mean of confusion matrices from within subject CV is: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
        #perc = np.asarray(confusion)/np.sum(confusion, 1).astype(float)
        logger.info("Mean percentage confusion matrix from within subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
        logger.info("Std of percentage confusion matrices from within subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



############ WITHIN 4 SUBJECT CV - 3FOLD FOR 4 SUBJECTS #############
for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], reliable_subj, k=10, mat=mat)
        logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
        logger.info("Results per fold: \n"+metrics.to_string())
        for k in range(len(confusion)):
            logger.info("Confusion matrices per folds are: \n"+pd.DataFrame(confusion[k]).to_string())
        logger.info("Mean of confusion matrices from within 4 subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
        logger.info("Mean percentage confusion matrix from within 4 subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
        logger.info("Std of percentage confusion matrices from within  4 subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



############ ACROSS 4 SUBJECT CV #############
for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = AcrossSubjectCV(estimators[i], reliable_subj, mat)
        logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
        logger.info("Results per fold: \n"+metrics.to_string())
        for k in range(len(confusion)):
            logger.info("Confusion matrices per folds are: \n"+pd.DataFrame(confusion[k]).to_string())
        logger.info("Mean of confusion matrices from across subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
        logger.info("Mean percentage confusion matrix from across subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
        logger.info("Std of percentage confusion matrices from across subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



