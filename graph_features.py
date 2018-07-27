import numpy as np
import pandas as pd
import networkx as nx
import os
from scipy.stats import ttest_1samp
from feature_extraction import extract_opt_2class
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from build_features import augment
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

gridsearch_scores = ['roc_auc','accuracy','f1']
best_score=False

cwd = os.getcwd()

def rebuild_symmetric(X_array,size=90):
    X_new = np.zeros((90,90))
    index = np.tril_indices(90)
    X_new[index]=X_array
    X_new = X_new + X_new.T
    di = np.diag_indices(90)
    X_new[di] = X_new[di]/2
    return X_new

def myGlobalEfficency(G,node):
    s = 0
    for j in G:
        if not j==node:
            s+= nx.efficiency(G, node, j)
    return s

def prepare_single_graph_features(X):
    """ X : connectivity array of shape[4095, nfreq].
    """
    S = []
    Div = []
    D = []
    #LocalEff = []
    GlobalEff = []
    for freq in range(len(X[0,:])):
        currentC = rebuild_symmetric(X[:, freq])
        currentA = np.nan_to_num(currentC/currentC)
        G = nx.from_numpy_matrix(currentA)
        currentC[np.diag_indices(90)] = 0
        S.append([np.mean(currentC[i,:])*(90/89) for i in G])
        Div.append([np.sum(np.delete((currentC[i,:]-S[-1])**2, i))/89 for i in G])
        D.append([nx.degree(G, i) for i in G])
        GlobalEff.append([myGlobalEfficency(G, i) for i in G])
    return (S, Div, D, GlobalEff)

def prepare_graph_features(name='one', p_threshold=0.05):
    """ X : connectivity array of shape[4095, nfreq].
    """
    X, Y  = extract_opt_2class(name, 'y', graph=True, p_threshold=p_threshold)
    n,_,nfreq = np.shape(X)
    R = np.zeros((n, 4, nfreq, 90))
    for i in range(n):
        if (i%100==0):
            print(i)
        res = prepare_single_graph_features(X[i])
        for j in range(4):
            R[i,j,:,:] = np.asarray(res[j])
    R = np.reshape(R, (n, -1))
    np.save(cwd+'/graph_features/'+'{}_tresh{}'.format(name, p_threshold), R)
    return(R, Y)


if __name__=="__main__":
    #prepare_graph_features()
    #prepare_graph_features('std')
    prepare_graph_features('one', 0.01)
    prepare_graph_features('std', 0.01)