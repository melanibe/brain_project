import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import time 
import argparse


############## PARAMS SETUP ###############
bestPCA_SVM = [ {'pca__n_components':  [50], 'svm__kernel': ['rbf']}]
bestRF = [{'PerBest__percentile': [100], 'rf__n_estimators': [10000], 'rf__min_samples_split':[30]}]
bestSVM = [ {'svm__kernel': ['rbf'], 'svm__C':[1]}]
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



############ LOADING DATA #############
print("The feature matrix is {}".format(type_agg))
try:
    X = np.load(cwd+"/{}.npy".format(type_agg))
    Y = np.load(cwd+"/y.npy")
except:
    print("Not found. Should prepare X and Y first.")
n,m = np.shape(X)

# distinguish only REM-nonREM
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=test_size, random_state=42, stratify=Y_main)

print(len(X_train))
neg_ix = np.where([Y_train[i]==0 for i in range(len(X_train))])[0]
pos_ix = np.where([Y_train[i]==1 for i in range(len(X_train))])[0]
aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix), replace=True)
train_ix = np.append(neg_ix, aug_pos_ix)
X_train = [X_train[i] for i in train_ix]
Y_train = [Y_train[i] for i in train_ix]
print(len(X_train))
print(len(np.where([Y_train[i]==0 for i in range(len(X_train))])[0]))
print(len(np.where([Y_train[i]==1 for i in range(len(X_train))])[0]))

################ GRIDSEARCH NORMALIZED + PCA + SVC ###############
pipePCA = Pipeline([ ('var', VarianceThreshold(threshold=0)),('pca', PCA(n_components = 500)), ('std', StandardScaler()), ('svm', SVC(kernel='linear', probability=True, verbose=True)) ])
pipePCA.fit(X_train, Y_train)
score = pipePCA.predict_proba(X_test)
pred = pipePCA.predict(X_test)
print(accuracy_score(Y_test,pred))
print(roc_auc_score(Y_test, score[:,1]))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,pred))