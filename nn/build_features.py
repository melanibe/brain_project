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
def prepare_X(): 
    """ loads all the original matlab coherence matrices in one big matrix of shape [10081, 4095, 50].
    """
    X = []
    Y = []
    i=0
    t1 = time.time()
    for subj in os.listdir(data_folder):
        print(subj)
        path_subj = os.path.join(data_folder, subj)
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

def index_subj(s): 
    """ loads all the original matlab coherence matrices in one big matrix of shape [10081, 4095, 50].
    """
    idx = []
    subj_idx = []
    i=0
    for subj in os.listdir(data_folder):
        print(subj)
        path_subj = os.path.join(data_folder, subj)
        if os.path.isdir(path_subj):
            for phase in os.listdir(path_subj):
                path_phase = os.path.join(path_subj, phase)
                if os.path.isdir(path_phase):
                    for file in os.listdir(path_phase):
                        test = re.search( r'average', file)
                        if test == None:
                            if not subj == s:
                                idx.append(i)
                            else:
                                subj_idx.append(i)
                            i+=1
    print(np.shape(idx))
    print(np.shape(subj_idx))
    return(idx, subj_idx)

def save_subj_list():
    res = pd.DataFrame()
    idx = []
    subj_idx = []
    for s in subject_list:
        tmp = index_subj(s)
        idx.append(tmp[0])
        subj_idx.append(tmp[1])
    res['subj_idx'] = subj_idx
    res['idx'] = idx
    res['subj'] = subject_list
    res.to_pickle(cwd+'/subj_list')


    
def prepare_X_bands():
    """ saves all the subset of the original matrix per standard frequency band.
    """
    X, Y = prepare_X()
    n,_,_ = np.shape(X)
    np.save(cwd+"/y", Y)
    np.save(cwd+"/X_delta", np.reshape(X[:,:,0:3], (n,-1))) #1 to 3 Hz
    np.save(cwd+"/X_theta",np.reshape(X[:,:,3:7], (n,-1))) #4 to 7
    np.save(cwd+"/X_alpha", np.reshape(X[:,:,7:12], (n,-1))) # 8 to 12
    np.save(cwd+"/X_beta", np.reshape(X[:,:,12:30], (n,-1)))  # 13 to 30
    np.save(cwd+"/X_gamma", np.reshape(X[:,:,30:], (n,-1)))  #over 31
    return(1)

def prepare_X_mine(index_list = [0,5,10,15], channels_per_freq=1000):
    """ Build the Just4Freq1000Channels matrix mentioned in the report.
    """
    try:
       X_new = np.load(cwd+"/X_mine_{}.npy".format(channels_per_freq))
       Y = np.load(cwd+"/y.npy")
       return(X_new, Y)
    except:
        print("Sorry have to prepare X_mine_{}".format(channels_per_freq))   
        X, Y = prepare_X()
        np.save(cwd+"/y", Y)
        tmp = X[:,:,index_list]
        vars = np.var(tmp, axis=0)
        n,m = np.shape(vars)
        idx = []
        for i in range(len(index_list)):
            l = np.argsort(vars[:, i])
            idx.append([np.ravel_multi_index((u,i), (n,m)) for u in l[-channels_per_freq:]]) #get list of indices in flattened way
        idx = np.reshape(idx, (-1))
        idx = np.unravel_index(idx, (n,m))
        X_new = X[:,idx[0], idx[1]]
        print(np.shape(X_new))
        np.save(cwd+"/X_mine_{}".format(channels_per_freq), X_new) 
        return(X_new, Y)

def prepare_X_std_freq():
    """ Build the matrix that aggregates the values per standard frequency band 
    i.e. Std matrix of the report.
    """
    X, Y = prepare_X()
    np.save(cwd+"/y", Y)
    X_delta = np.mean(X[:,:,0:3], axis=2) #1 to 3 Hz
    print(np.shape(X_delta))
    X_theta = np.mean(X[:,:,3:7], axis=2) #4 to 7 Hz
    X_alpha = np.mean(X[:,:,7:13], axis=2) #8-13 Hz
    X_beta = np.mean(X[:,:,13:30], axis=2) #14-30 Hz
    X_gamma = np.mean(X[:,:,30:], axis=2) #>30 Hz
    print(np.shape(X_gamma))
    X_aggregated = np.concatenate((X_delta, X_theta, X_alpha, X_beta, X_gamma), axis =1)
    np.save(cwd+"/std", X_aggregated)
    print(np.shape(X_aggregated))
    return (X_aggregated,Y)

def prepare_X_one_freq():
    """ Prepare the matrix that aggregates the values over all frequencies.
    i.e. One matrix in the report.
    """
    X, Y = prepare_X()
    np.save(cwd+"/y", Y)
    X_one = np.mean(X[:,:,:], axis=2)
    np.save(cwd+"/one", X_one)
    print(np.shape(X_one))
    return (X_one, Y)    

def prepare_X_ten_freq(): 
    """ DEPRECATED.
    Averaged over 10 freqeuncy bins.
    """
    X, Y = prepare_X()
    np.save(cwd+"/y", Y)
    l = []
    i = 0
    while (i<50):
        l.append(np.mean(X[:,:,i:i+5], axis=2))
        i = i+5
    X_ten = np.concatenate(l,axis=1)
    np.save(cwd+"/ten", X_ten)
    print(np.shape(X_ten))
    return (X_ten, Y)  

def prepare_X_std_freq_channels(channels_per_freq=200):
    """ DEPRECATED.
    Builds the std matrix and then keeps only the channels_per_freq most varying channels per 
    frequency band.
    """
    try:
        X_new = np.load(cwd+"/X_std_{}channels.npy".format(channels_per_freq))
        Y = np.load(cwd+"/y.npy")
        return (X_new,Y)
    except:
        print("Have to prepare X_std_{}channels.npy".format(channels_per_freq))
        try:
            X_aggregated = np.load(cwd+"/X_std.npy")
            Y = np.load(cwd+"/y")
        except:
            X_aggregated, Y = prepare_X_std_freq()
        X_aggregated = np.reshape(X_aggregated, (-1,4095,5))
        _, _, c = np.shape(X_aggregated)
        print(np.shape(X_aggregated))
        tmp = X_aggregated
        vars = np.var(tmp, axis=0)
        n, m = np.shape(vars)
        idx = []
        for i in range(c):
            l = np.argsort(vars[:, i])
            idx.append([np.ravel_multi_index((u,i), (n,m)) for u in l[-channels_per_freq:]]) #get list of indices in flattened way
        idx = np.reshape(idx, (-1))
        idx = np.unravel_index(idx, (n,m))
        X_new = np.reshape(X_aggregated[:,idx[0], idx[1]],(-1,channels_per_freq*5))
        print(np.shape(X_new)) 
        np.save(cwd+"/X_std_{}channels".format(channels_per_freq), X_new)
        return (X_new,Y)


def augment(X, Y, save=False, name=None):
    """ Performs data augmentation against class imbalance.
    """
    idx = [True if ((y==1) or (y==2)) else False for y in Y]
    X_new = np.concatenate((X, X[idx,:]))
    print(np.shape(X_new))
    Y_new = np.concatenate((Y, np.repeat(1, np.sum(idx))))
    print(np.shape(Y_new))
    if save:
        np.save(cwd+'/{}_aug'.format(name), X_new)
        np.save(cwd+'/Y_aug', Y_new)
    return(X_new, Y_new)


# for testing
if __name__=='__main__':
    #prepare_X_std_freq()
    #prepare_X_one_freq()
    #X = np.load(cwd+'/X_delta.npy')
    #y = np.load(cwd+'/Y.npy')
    #c, _ = augment(X,y)
    #print(np.shape(c))
    #index_subj('S01')
    save_subj_list()
    p = pd.read_pickle(cwd+'/subj_list')
    print(p.columns)
    l = p.loc[p['subj']=='S01', 'idx'].values[0]
    print(l[0])
    l = p.loc[p['subj']=='S01', 'subj_idx'].values[0]
    print(l[0])
