import logging
import time
import os 
import torch
import numpy as np

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import GraphConvNet
from utils import my_eval

run_number = '_21Jul18_154207'
batch_size = 32

cwd = os.getcwd()
try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project")
    cwd = os.getcwd()
except: # for cluster run
    cwd = os.getcwd()
checkpoint_dir = cwd + '/nn/runs/'

# ------------------------ Load data --------------------- #
X = np.load(cwd+'/./one.npy')
Y = np.load(cwd+'/./y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)

# Creating the batches (balanced classes)
X_testloader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=4)
Y_testloader = torch.utils.data.DataLoader(Y_test, batch_size=batch_size, shuffle=False, num_workers=4)

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
    for xtest, ytest in zip(X_testloader, Y_testloader):
        if len(ytest)==batch_size:
            y_really_test.append(ytest)
            outputs = gcn(xtest)
            proba.append(outputs.data.numpy())
            _, predicted = torch.max(outputs.data, 1)
            total += ytest.size(0)
            correct += (predicted == ytest).sum().item()
            c = (predicted == ytest).squeeze()
            for i in range(len(ytest)): # change that (euh i can't remember what to change)
                label = ytest[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
for i in range(2):
    print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))

print('Accuracy test : %d %%' % (100 * correct / total))
## Get ROC score
print('ROC is {} '.format(roc_auc_score(np.concatenate(y_really_test), np.concatenate(proba)[:,1])))

