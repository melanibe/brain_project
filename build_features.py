import scipy.io as sio
import os
import re
import numpy as np
import time
import pandas as pd

""" File builds the feature matrices.
"""

try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project/")
    cwd = os.getcwd()
    data_folder = cwd + '/DATA/'
except: # for cluster
    cwd = os.getcwd()
    data_folder = "/cluster/scratch/melanibe/"+'/DATA/'

phases = {"REM_phasic":1,"REM_tonic":2,"S2_Kcomplex":3,"S2_plain":4,"S2_spindle":5,"SWS_deep":6}
subject_list = ['S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S11', 'S12']
################## LOADING DATA AND PREPARING THE MATRIX ##################
def prepare_X(subject_list= subject_list): 
    """ loads all the original matlab coherence matrices in one big matrix of shape [10081, 4095, 50].
    """
    X = []
    Y = []
    i=0
    t1 = time.time()
    for subj in os.listdir(data_folder):
        path_subj = os.path.join(data_folder, subj)
        if subj in subject_list:
            print(subj)
            if os.path.isdir(path_subj):
                for phase in os.listdir(path_subj):
                    path_phase = os.path.join(path_subj, phase)
                    if os.path.isdir(path_phase):
                        for file in os.listdir(path_phase):
                            path_file = os.path.join(path_phase, file)
                            test = re.search( r'average', file)
                            if test == None:
                                X.append(np.reshape(np.asarray(sio.loadmat(path_file)['TF']),(4095,50)))
                                Y.append(phases[phase])
                            i+=1
    t2 = time.time()
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(np.shape(X))
    print(np.shape(Y))
    print(t2-t1)
    return(X,Y)

def transform_X_std(X):
    """ Build the matrix that aggregates the values per standard frequency band 
    i.e. Std matrix of the report.
    """
    X_delta = np.mean(X[:,:,0:3], axis=2) #1 to 3 Hz
    print(np.shape(X_delta))
    X_theta = np.mean(X[:,:,3:7], axis=2) #4 to 7 Hz
    X_alpha = np.mean(X[:,:,7:13], axis=2) #8-13 Hz
    X_beta = np.mean(X[:,:,13:30], axis=2) #14-30 Hz
    X_gamma = np.mean(X[:,:,30:], axis=2) #>30 Hz
    print(np.shape(X_gamma))
    X_aggregated = np.concatenate((X_delta, X_theta, X_alpha, X_beta, X_gamma), axis =1)
    print(np.shape(X_aggregated))
    return (X_aggregated)

def transform_X_one(X):
    """ Prepare the matrix that aggregates the values over all frequencies.
    i.e. One matrix in the report.
    """
    X_one = np.mean(X[:,:,:], axis=2)
    print(np.shape(X_one))
    return (X_one)    


# for testing
if __name__=='__main__':
    # creating the sub-directories to save the matrices if necessary
    dir = cwd+'/matrices/std/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created '+  dir)
    dir = cwd+'/matrices/one/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created '+  dir)
    dir = cwd+'/matrices/all/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created '+  dir)
    for s in subject_list:
        print(s)
        X, Y = prepare_X([s])
        Xstd = transform_X_std(X)
        Xone = transform_X_one(X)
        print(np.shape(Xstd))
        np.save(cwd+'/matrices/all/{}'.format(s), X)
        np.save(cwd+'/matrices/std/{}'.format(s), Xstd)
        np.save(cwd+'/matrices/one/{}'.format(s), Xone)
        np.save(cwd+'/matrices/{}_y'.format(s), Y)
