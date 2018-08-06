import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
cwd = os.getcwd()
long_subject_list = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']

class UpsampleStratifiedKFold:
    """ custom cv generator for upsampling.
    """
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X,y):
            neg_ix = np.where([y[i]==0 for i in train_idx])[0]
            neg_ix = [train_idx[i] for i in neg_ix]
            pos_ix = np.where([y[i]==1 for i in train_idx])[0]
            aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix), replace=True)
            aug_pos_ix = [train_idx[i] for i in aug_pos_ix]
            train_idx = np.append(neg_ix, aug_pos_ix)
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


###### ACROSS CV #####
def AcrossSubjectCV(estimator, logger, subject_list=long_subject_list, mat = 'std', upsample=False):
    """ Implements custom CV to calculate the across subject accuracy.
    10-fold CV since we have 10 subjects.
    """   
    roc_auc=[]
    accuracyCV = []
    confusion = []
    conf_perc = []
    for s in subject_list:
        X_test, Y_test = np.load(cwd+'/matrices/{}_{}.npy'.format(s, mat)), np.load(cwd+'/matrices/{}_y.npy'.format(s))
        Y_test = [1 if ((y==1) or (y==2)) else 0 for y in Y_test]
        print("Preparing fold only {}".format(s))
        print("Preparing fold except {}".format(s))
        first=True
        for other in subject_list:
            if not other == s:
                print(other)
                if first:
                    X_train = np.load(cwd+'/matrices/{}_{}.npy'.format(other, mat))
                    Y_train = np.load(cwd+'/matrices/{}_y.npy'.format(other))
                    first=False
                else:
                    X_train = np.concatenate((X_train, np.load(cwd+'/matrices/{}_{}.npy'.format(other, mat))), axis =0)
                    Y_train = np.concatenate((Y_train, np.load(cwd+'/matrices/{}_y.npy'.format(other))), axis =0)        
        Y_train = [1 if ((y==1) or (y==2)) else 0 for y in Y_train]
        n = len(X_train)
        print(n)
        print(len(X_test))
        print(n+len(X_test))
        if upsample:
        # perform upsampling only on training fold
            neg_ix = np.where([Y_train[i]==0 for i in range(n)])[0]
            pos_ix = np.where([Y_train[i]==1 for i in range(n)])[0]
            assert(len(pos_ix)+len(neg_ix)==n)
            aug_pos_ix = np.random.choice(pos_ix, size=len(neg_ix), replace=True)
            X_train = np.reshape(np.append([X_train[i] for i in neg_ix], [X_train[i] for i in aug_pos_ix]),(len(neg_ix)+len(aug_pos_ix),-1))
            Y_train = np.append([0 for i in neg_ix], [1 for i in aug_pos_ix])
        n = len(X_train)
        logger.info("REM/nonREM ratio current fold: {}".format(np.sum([Y_train[i]==1 for i in range(n)])/float(np.sum([Y_train[i]==0 for i in range(n)]))))
        print("Fit the estimator")
        try: #GCN fit takes X_val and Y_val in arguments but not the others
            estimator.fit(X_train, Y_train, X_test, Y_test, "_across_testsubj_{}".format(s))
        except:
            estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, pred))
        conf = confusion_matrix(Y_test, pred)
        confusion.append(conf)
        true_freq = np.reshape(np.repeat(np.sum(conf, 1),2,axis=0), (2,2))
        conf_perc.append(np.nan_to_num(conf.astype(float)/true_freq))
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
    return(results, metrics, confusion, conf_perc)



def WithinOneSubjectCV(estimator, logger, subject = ['S12','S10','S12','S05'], k=10, upsample=False, mat = 'std'):
    """ Implements custom CV within one or more subject, 3-fold CV
    """
    roc_auc=[]
    accuracyCV = []
    confusion = []
    conf_perc = []
    first=True
    for s in subject:
        if first:
            X = np.load(cwd+'/matrices/{}_{}.npy'.format(s, mat))
            Y = np.load(cwd+'/matrices/{}_y.npy'.format(s))
            first=False
        else:
            X = np.concatenate((X, np.load(cwd+'/matrices/{}_{}.npy'.format(s, mat))), axis =0)
            Y = np.concatenate((Y, np.load(cwd+'/matrices/{}_y.npy'.format(s))), axis =0)        
    Y = [1 if ((y==1) or (y==2)) else 0 for y in Y]
    print(np.shape(X))
    if upsample:
        print('up')
        cv_gen = UpsampleStratifiedKFold(k)
    else:
        cv_gen = StratifiedKFold(k, shuffle=True, random_state=42)
    nsubj = len(subject)
    fold = 0
    for train_index, test_index in cv_gen.split(X, Y):
        fold +=1
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
        # check proportion
        logger.info("REM/nonREM ratio current train fold: {}".format(np.sum([Y_train[i]==1 for i in range(len(X_train))])/float(np.sum([Y_train[i]==0 for i in range(len(X_train))]))))
        logger.info("REM/nonREM ratio current test fold: {}".format(np.sum([Y_test[i]==1 for i in range(len(X_test))])/float(np.sum([Y_test[i]==0 for i in range(len(X_test))]))))
        try: #GCN fit takes X_val and Y_val in arguments but not the others
            if nsubj==1: # for the file names
                estimator.fit(X_train, Y_train, X_test, Y_test, "_within_subj_{}_fold_{}".format(subject[0], fold))
            else: # for the file names
                estimator.fit(X_train, Y_train, X_test, Y_test, "_within_{}_subjects_fold_{}".format(nsubj, fold))
        except:
            estimator.fit(X_train, Y_train)
        print("Calculating the metrics")
        pred = estimator.predict(X_test)
        roc_auc.append(roc_auc_score(Y_test, estimator.predict_proba(X_test)[:,1]))
        accuracyCV.append(accuracy_score(Y_test, pred))
        conf = confusion_matrix(Y_test, pred)
        confusion.append(conf)
        true_freq = np.reshape(np.repeat(np.sum(conf, 1),2,axis=0), (2,2))
        conf_perc.append(np.nan_to_num(conf.astype(float)/true_freq))
        print(roc_auc[-1], accuracyCV[-1])
    metrics = pd.DataFrame()
    metrics['auc'] = roc_auc
    metrics['accuracy'] = accuracyCV
    results = pd.DataFrame()
    results['mean'] = [np.mean(roc_auc), np.mean(accuracyCV)]
    results['std'] = [np.std(roc_auc), np.std(accuracyCV)]
    results['min'] = [np.min(roc_auc), np.min(accuracyCV)]
    results['max'] = [np.max(roc_auc), np.max(accuracyCV)]
    results.index=['roc_auc', 'accuracy']
    return(results, metrics, confusion, conf_perc)

