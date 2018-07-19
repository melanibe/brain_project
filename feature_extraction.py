import numpy as np
import os
from build_features import prepare_X
from scipy.stats import ttest_ind

""" Implements the feature extraction algorithm of the paper.
"""

cwd = os.getcwd()

def extract_opt_2class(name='std', y = 'y', graph=False):
    if name=='X':
        X,Y = prepare_X()
    else:
        Y = np.load(cwd+'/{}.npy'.format(y))
        X = np.reshape(np.load(cwd+'/{}.npy'.format(name)),(-1,4095,5))
    m, _, nfreq = np.shape(X)
    Y = [1 if ((y==1) or (y==2)) else 0 for y in Y]
    idx0 = [False if ((y==1) or (y==2)) else True for y in Y]
    idx1 = [True if ((y==1) or (y==2)) else False for y in Y]
    X0 = X[idx0,:,:]
    X1 = X[idx1,:,:]
    print(np.shape(X1))
    print(np.shape(X0))
    l= []
    for s in range(4095):
        if s%100==0:
            print(s)
        for f in range(nfreq):
            _, p = ttest_ind(X0[:,s,f], X1[:,s,f],equal_var=False)
            if p < 0.05:
                l.append(np.ravel_multi_index((s,f), (4095,nfreq)))
    n= len(l)
    print(len(l))
    l = np.reshape(l, (-1))
    l = np.unravel_index(l, (4095,nfreq))
    if graph:
        X_new = np.zeros((m, 4095, nfreq))
        X_new[:,l[0], l[1]] = X[:,l[0], l[1]]
        return(X_new)
    else:
        np.save(cwd+'/{}_sel'.format(name), np.reshape(X[:,l[0], l[1]],(-1,n)))



if __name__=='__main__':
    extract_opt_2class()
    # print(np.shape(np.load(cwd+'/std_sel.npy')))
    extract_opt_2class('std_aug', 'Y_aug')
    extract_opt_2class('X')