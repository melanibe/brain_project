import numpy as np
import pandas as pd
from build_features import prepare_X_without_subj, prepare_X_just_subj
from sklearn.metrics import roc_auc_score, accuracy_score

subject_list = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']

def AcrossSubjectCV(estimator):
    """ Implements custom CV to calculate the across subject accuracy.
    10-fold CV since we have 10 subjects.
    """
    roc_auc=[]
    accuracyCV = []
    for s in subject_list:
        print("Preparing fold only {}".format(s))
        X_test, Y_test = prepare_X_just_subj(s)
        print("Preparing fold except {}".format(s))
        X_train, Y_train = prepare_X_without_subj(s)
        print("Fit the estimator")
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


        


