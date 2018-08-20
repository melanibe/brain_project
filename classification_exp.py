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

""" TO DO:
add parser argument for GCN parameter
could also add parser argument to choose the type of cross val to run
"""

""" 
Runs the validation runs of classification report. For now only to use with std, other matrices not built.
"""

############## PARAMS SETUP ###############
parser = argparse.ArgumentParser()

parser.add_argument("-est", \
                    "--estimatorlist", \
                    nargs='*', \
                    help="list of estimator among uniform constant, gcn, pcasvm, rf")

parser.add_argument("-s", "--nsteps", help="number of steps for gcn training", type=int)
parser.add_argument("-up", "--upsample", help="if you want upsampling in the CVs", type=bool)
parser.add_argument("-j", "--njobs", help="number of jobs for sklearn", type=int)
parser.add_argument("-t", "--type", help="choose the preprocessing, one or std aggregation")
args = parser.parse_args()

if args.type:
    mat = args.type
else:
    mat = 'std'

if args.upsample:
    upsample = args.upsample
else:
    upsample = False

if args.njobs:
    njobs = args.njobs
else:
    njobs = 3

if args.nsteps:
   nsteps = args.nsteps
else:
    nsteps = 300

try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project/")
    cwd = os.getcwd()
except: # for cluster run
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

if upsample:
    logger.warning("Using upsampling")

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
                         ('rf', RandomForestClassifier(n_estimators=10000, min_samples_split=30, n_jobs=4))])

GCN_estimator = GCN_estimator_wrapper(checkpoint_dir, logger, 64, 32, 16, nsteps = nsteps, reset=True)


args_to_est = {'uniform': uniform, 'constant': constant, 'pcasvm': pipePCA_SVM, 'rf': pipeKBest_RF, 'gcn': GCN_estimator}

try:
    if args.estimatorlist:
        print(args.estimatorlist)
        estimators=[]
        for a in args.estimatorlist:
            estimators.append(args_to_est[a])
        names = args.estimatorlist
        print(estimators)
    else:
        estimators = [GCN_estimator, pipePCA_SVM, pipeKBest_RF]
        names = ['GCN',  'PCA and SVM', 'KBest and RF']
except:
    logger.error("You provided a wrong argument for estimator")



############ WITHIN ONE SUBJECT CV - 5-FOLD FOR 4 SUBJECTS #############
reliable_subj = ['S12', 'S10', 'S04', 'S05']

for subject in reliable_subj:
    for i in range(len(estimators)):
        print(subject)
        results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], logger, subject=[subject], k=4, mat=mat, upsample=upsample)
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
    results, metrics, confusion, conf_perc = WithinOneSubjectCV(estimators[i], logger, reliable_subj, k=10, mat=mat, upsample=upsample)
    logger.debug("Results per fold: \n"+metrics.to_string())
    for k in range(len(confusion)):
        logger.debug("Confusion matrices fold {} is: \n".format(k)+pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from within 4 subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from within 4 subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
    logger.info("Std of percentage confusion matrices from within  4 subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



############ ACROSS 4 SUBJECT CV #############
for i in range(len(estimators)):
    logger.info("Results for across 4 subject CV for {} estimator".format(names[i]))
    results, metrics, confusion, conf_perc = AcrossSubjectCV(estimators[i], logger, reliable_subj, mat, upsample=upsample)
    logger.info("Results per fold: \n"+metrics.to_string())
    for k in range(len(confusion)):
        logger.info("Confusion matrices per folds are: \n"+pd.DataFrame(confusion[k]).to_string())
    logger.info("Mean results accross folds for {} estimator with subject CV are: \n".format(names[i])+results.to_string())
    logger.info("Mean of confusion matrices from across subject CV are: \n {} \n".format(pd.DataFrame(np.mean(confusion, 0)).to_string()))
    logger.info("Mean percentage confusion matrix from across subject CV \n: {}".format(pd.DataFrame(np.mean(conf_perc,0))))
    logger.info("Std of percentage confusion matrices from across subject CV is: \n {} \n".format(pd.DataFrame(np.std(conf_perc, 0)).to_string()))



