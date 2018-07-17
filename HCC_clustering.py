import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import parmap
import argparse

""" This file implements the HCC clustering of the coherence matrices. 
For now using mean(coherence) as similarity measure b/w cluster.
Should implement the cluster similarity measure of the paper later maybe

IDEA/TO-DO
Run the clsutering on every coherence matrice and use the cluster number associated to each region as a predictor.
Would give a [10081, 90] feature matrix. 
Try several number of clusters and different cluster similarity measure (mean, min, max, CCo)  

NOTE
Assumes that the frequency band matrices are already constructed and saved. (Do wanna build them each time since it
takes too much time).
"""

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--njobs", help="number of jobs", type=int)
parser.add_argument("-t", "--type", help="choose the type clustering")
parser.add_argument("-b", "--band", help="frequency band") # delta 0:4, that 4:8, alpha 8:12, beta 12:30, gamma 30-50
parser.add_argument("-c", "--clusters", help="number of clusters", type=int)
args = parser.parse_args()

type = args.type
band = args.band
n_cluster = args.clusters

if args.njobs:
    njobs = args.njobs
else:
    njobs = 1


try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/RESEARCH_PROJECT/")
    cwd = os.getcwd()
    data_folder = cwd + '/DATA/'
except: # for cluster run
    cwd = os.getcwd()
    data_folder = "/cluster/scratch/melanibe/"+'/DATA/'

def rebuild_symmetric(X_array,size=90):
    X_new = np.zeros((90,90))
    index = np.tril_indices(90)
    X_new[index]=X_array
    X_new = X_new + X_new.T
    di = np.diag_indices(90)
    X_new[di] = X_new[di]/2
    return X_new



def HCC(obs, n_cluster, C_band, type='mean'):
    n,_ = np.shape(C_band)
    C = np.reshape(C_band, (n,4095,-1))
    C_new = np.mean(C, axis=2) # averaged over frequency band
    C_new = rebuild_symmetric(C_new[obs])
    D = 1-C_new
    k = 90
    min_value = []
    clusters = np.arange(1,91)
    min_value.append(np.min(D))
    i, j = np.unravel_index(np.argmin(D), D.shape)
    if i<j:
        clusters[j]=clusters[i]
        clusters[(j+1):]=clusters[(j+1):]-1
    else:
        clusters[i]=clusters[j]
        clusters[(i+1):]=clusters[(i+1):]-1
    k -=1
    while (k>n_cluster):
        c1, c2 = 0, 1
        d_min = 1
        for i in range(1, k):
            for j in range(2, k+1):
                tmp = C_new[clusters==i,:]
                if len(tmp)==0:
                    print('error')
                tmp = tmp[:,clusters==j]
                if type=='mean':
                    d = 1-np.mean(tmp)
                elif type=='min':
                    d = 1-np.min(tmp)
                elif type=='max':
                    d = 1-np.max(tmp)
                if d < d_min:
                    d_min = d
                    c1, c2 = i, j
        clusters[clusters == c2]=c1
        clusters[clusters>c2]=clusters[clusters>c2]-1
        min_value.append(d_min)
        k-=1
    return(clusters)


if __name__=="__main__": # it is reaaallly slow but with parmap packages memory error on the cluster so no // for now.
    C = np.load(cwd+'/X_{}.npy'.format(band))
    n, _ = np.shape(C)
    Xclusters=np.zeros((n,90))
    for i in range(n):
        if i%10==0:
            print(i)
        Xclusters[i,:]= HCC(i, n_cluster, C, type)
    np.save(cwd+'/cluster_{}_{}_{}'.format(band, type, n_cluster) , Xclusters)



