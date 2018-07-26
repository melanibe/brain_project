import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import logging
import time 
import argparse

from CV_utils import AcrossSubjectCV, WithinOneSubjectCV

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



######### best with graph
estimator_graph = Pipeline([('var', VarianceThreshold(threshold=0)), \
                            ('std', StandardScaler()), \
                            ('PerBest', SelectPercentile(percentile=50)), \
                            ('rf', RandomForestClassifier(n_estimators=10, min_samples_split=30, n_jobs=njobs))])        
matrix_graph = np.load(cwd+'/graph_features/std_tresh0.05.npy')
Y = np.load(cwd+'/y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
list = ['S04', 'S10', 'S12', 'S05']
tmp = AcrossSubjectCV(estimator_graph, matrix_graph, Y_main, list )
#tmp = WithinOneSubjectCV(estimator_graph, matrix_graph, Y_main)
logger.info("Global results for best graph feature estimator from accross subject CV are: \n"+tmp[0].to_string())
print(tmp[1])
for i in range(len(list)):
    logger.info("Confusion matrix for subject {} as test is: \n".format(list[i])+pd.DataFrame(tmp[2][i]).to_string())
logger.info("Mean of confusion matrices from accross subject CV are: \n {}".format(pd.DataFrame(np.mean(tmp[2], 0)).to_string()))
logger.info("Results per subject for best graph feature estimator from accross subject CV are: \n"+tmp[1].to_string())


########### best with orig
matrix_full = np.load(cwd+'/X_sel.npy')
estimator_full = Pipeline([('var', VarianceThreshold(threshold=0)), \
                            ('std', StandardScaler()), \
                            ('PerBest', SelectPercentile(percentile=10)), \
                            ('rf', RandomForestClassifier(n_estimators=10000, min_samples_split=30, n_jobs=njobs))])
tmp = AcrossSubjectCV(estimator_full, matrix_full, Y_main)
logger.info("Global results for best graph feature estimator from accross subject CV are: \n"+tmp[0].to_string())
logger.info("Results per subject for best graph feature estimator from accross subject CV are: \n"+tmp[1].to_string())