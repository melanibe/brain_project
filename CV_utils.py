import numpy as np
import pandas as pd
import os

from build_features import index_subj
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, RepeatedStratifiedKFold

cwd = os.getcwd()
long_subject_list = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']

###### ACROSS CV #####
def AcrossSubjectCV(estimator, X, Y, subject_list=long_subject_list):
    """ Implements custom CV to calculate the across subject accuracy.
    10-fold CV since we have 10 subjects.
    """
    p = pd.read_pickle(cwd+'/subj_list')
    roc_auc=[]
    accuracyCV = []
    confusion = []
    for s in subject_list:
        _, subj_idx = p.loc[p['subj']==s, 'idx'].values[0], p.loc[p['subj']==s, 'subj_idx'].values[0]
        print("Preparing fold only {}".format(s))
        X_test, Y_test = [X[i] for i in subj_idx], [Y[i] for i in subj_idx]
        print("Preparing fold except {}".format(s))
        other_idx = []
        for other in subject_list:
            if not other == s:
                other_idx += p.loc[p['subj']==other, 'subj_idx'].values[0]
        X_train, Y_train = [X[i] for i in other_idx], [Y[i] for i in other_idx]
        n = len(X_train)
        print(n)
        print(len(X_test))
        #assert(len(X_train)+len(X_test)==10081)
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
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, pred))
        confusion.append(confusion_matrix(Y_test, pred))
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
    return(results, metrics, confusion)



def OneSubjectCV(estimator, X, Y, subject = 'S12'):
    """ Implements custom CV within one subject, 3-fold CV
    """
    p = pd.read_pickle(cwd+'/subj_list')
    roc_auc=[]
    accuracyCV = []
    confusion = []
    subj_idx = p.loc[p['subj']==subject, 'subj_idx'].values[0]
    X, Y =  [X[i] for i in subj_idx], [Y[i] for i in subj_idx]
    Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
    X_train, X_test, Y_train, Y_test = StratifiedKFold()

    ##### TO DO THIS IS JUST COPY PASTE - USE UPSAPLING CV


    for s in subject_list:
        _, subj_idx = p.loc[p['subj']==s, 'idx'].values[0], p.loc[p['subj']==s, 'subj_idx'].values[0]
        print("Preparing fold only {}".format(s))
        X_test, Y_test = [X[i] for i in subj_idx], [Y[i] for i in subj_idx]
        print("Preparing fold except {}".format(s))
        other_idx = []
        for other in subject_list:
            if not other == s:
                other_idx += p.loc[p['subj']==other, 'subj_idx'].values[0]
        X_train, Y_train = [X[i] for i in other_idx], [Y[i] for i in other_idx]
        n = len(X_train)
        print(n)
        print(len(X_test))
        #assert(len(X_train)+len(X_test)==10081)
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
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, pred))
        confusion.append(confusion_matrix(Y_test, pred))
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
    return(results, metrics, confusion)