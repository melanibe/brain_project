import numpy as np
import networkx as nx
import os
from scipy.stats import ttest_1samp
from feature_extraction import extract_opt_2class
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

def prepare_single_graph_features(X, tr):
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
        Div.append([np.delete((currentC[i,:]-S[-1])**2, n)/89 for i in G])
        D.append([nx.degree(G, i) for i in G])
        GlobalEff.append([myGlobalEfficency(G, i) for i in G])
    return (S, Div, D, GlobalEff)


if __name__=="__main__":
    X, Y  = extract_opt_2class('std', 'y', graph=True)
    for i in len(2):
        print(i)
        v = prepare_single_graph(X[i])
        print(v)
