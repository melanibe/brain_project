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

# creating the directory for the run  
time.time()
t = time.strftime('%d%b%y_%H%M%S')
checkpoint_dir = cwd+'/runs/'+t +'/'
os.makedirs(checkpoint_dir)
print('Saving to '+  checkpoint_dir)


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
LOG_FILENAME= checkpoint_dir + '{}_classification_'.format(mat)+ t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)



############### CLASSIFIERS TO EVALUATE ############
#uniform = DummyClassifier(strategy='uniform')
#constant = DummyClassifier(constant=0, strategy='constant')
pipePCA_SVM = Pipeline([('var', VarianceThreshold(threshold=0)),\
                            ('pca', PCA(n_components=500)),\
                            ('std', StandardScaler()), \
                            ('svm', SVC(kernel='linear', probability=True))])
pipeKBest_RF = Pipeline([('var', VarianceThreshold(threshold=0)), \
                        ('std', StandardScaler()), \
                        ('PerBest', SelectPercentile(percentile=50)),\
                         ('rf', RandomForestClassifier(n_estimators=10000, min_samples_split=30, n_jobs=4))])
GCN_estimator = GCN_estimator_wrapper(checkpoint_dir, logger, 32, 64, 128, nsteps = 1000, reset=True)
#GCN_estimator = GCN_estimator_wrapper(checkpoint_file, logger, 256, 128, 128, batch_size= 128, reset=True)


############ WITHIN ONE SUBJECT CV - 5-FOLD FOR 4 SUBJECTS #############
reliable_subj = ['S12', 'S10', 'S04', 'S05']
#estimators = [GCN_estimator, pipePCA_SVM, pipeKBest_RF]
#names = ['GCN',  'PCA and SVM', 'KBest and RF']

estimators = [GCN_estimator]
names = ['GCN']

for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], logger, subject=[subject], k=4, mat=mat)
        logger.info("Results for subject {} for estimator {}".format(subject, names[i]))
        logger.debug("Results by fold: \n"+metrics.to_string())
        for k in range(len(confusion)):
            logger.debug("Confusion matrices fold {} is: \n".format(k)+pd.DataFrame(confusion[k]).to_string())
        logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
        logger.info("Mean of confusion matrices from within subject CV is: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
        logger.info("Mean percentage confusion matrix from within subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
        logger.info("Std of percentage confusion matrices from within subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



############ WITHIN 4 SUBJECT CV - 3FOLD FOR 4 SUBJECTS #############
for i in range(len(estimators)):
    logger.info("Result for within 4 subject (mixed) CV for {} estimator".format(names[i]))
    results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], logger, reliable_subj, k=10, mat=mat)
    logger.debug("Results per fold: \n"+metrics.to_string())
    for k in range(len(confusion)):
        logger.debug("Confusion matrices fold {} is: \n".format(k)+pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from within 4 subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from within 4 subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
    logger.info("Std of percentage confusion matrices from within  4 subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



############ ACROSS 4 SUBJECT CV #############
#estimators = [GCN_estimator, pipePCA_SVM, pipeKBest_RF]
#names = ['GCN',  'PCA and SVM', 'KBest and RF']
for i in range(len(estimators)):
    logger.info("Results for across 4 subject CV for {} estimator".format(names[i]))
    results, metrics, confusion, conf_perc = AcrossSubjectCV(estimators[i], logger, reliable_subj, mat, upsample=False)
    logger.info("Results per fold: \n"+metrics.to_string())
    for k in range(len(confusion)):
        logger.info("Confusion matrices per folds are: \n"+pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from across subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from across subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
    logger.info("Std of percentage confusion matrices from across subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



