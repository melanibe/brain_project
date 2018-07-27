import logging
import time
import os 
import torch
import numpy as np
import argparse


from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import GraphConvNetwork
from utils import my_eval, build_onegraph_A, ToTorchDataset
from scipy.linalg import block_diag

run_number = '24Jul18_142221'
batch_size = 64

cwd = os.getcwd()
try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project")
    cwd = os.getcwd()
except: # for cluster run
    cwd = os.getcwd()
checkpoint_dir = cwd + '/siamese_gcn/runs/'

# ------------------------ Load data --------------------- #
X = np.load(cwd+'/./std.npy')
Y = np.load(cwd+'/./y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)

test = ToTorchDataset(X_test, Y_test)
# Creating the batches (balanced classes)
X_testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=4)

# loading the model
gcn = torch.load(checkpoint_dir + run_number + '.pt')
print('Loaded model')


# evaluate model (surely can reuse eval helper so that it is not so messy.)
correct = 0
total = 0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
proba = list([])
y_really_test = []
with torch.no_grad():
    for data in X_testloader:
        labels = data['Y']
        coh_array1 = data['f1']
        coh_array2 = data['f2']
        coh_array3 = data['f3']
        coh_array4 = data['f4']
        coh_array5 = data['f5']
        #coh_array, labels = coh_array.to(device), labels.to(device)
        n, m = coh_array1.size()
        A1 = torch.zeros((n, 90, 90))
        A2 = torch.zeros((n, 90, 90))
        A3 = torch.zeros((n, 90, 90))
        A4 = torch.zeros((n, 90, 90))
        A5 = torch.zeros((n, 90, 90))
        # we don't have feature so use identity for each graph
        X = torch.eye(90).expand(n, 90, 90)
        for i in range(n):
            A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
            A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
            A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
            A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
            A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))
            #print(A)  
        y_really_test.append(labels)
        outputs = gcn(X, A1, A2, A3, A4, A5)
        proba.append(outputs.data.numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(len(labels)): # change that (euh i can't remember what to change)
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))

print('Accuracy test : %d %%' % (100 * correct / total))
## Get ROC score
print('ROC is {} '.format(roc_auc_score(np.concatenate(y_really_test), np.concatenate(proba)[:,1])))

