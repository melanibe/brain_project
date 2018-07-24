import logging
import time
import os 

import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag

from model import GraphConvNetwork
from utils import my_eval, build_onegraph_A, ToTorchDataset


# ----------- To allow run on GPU ------------ #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------------ Global parameters ------------- #

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", help="batch_size", type=int)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int)
parser.add_argument("-h1", "--h1", help="size hidden 1", type=int)
parser.add_argument("-h2", "--h2", help="size hidden 2", type=int)
parser.add_argument("-out", "--out", help="size out feat", type=int)
parser.add_argument("-lr", "--learning", help="learning rate", type=float)
args = parser.parse_args()

if args.batch:
    batch_size = args.batch
else:
    batch_size = 100

if args.epochs:
    n_epochs = args.epochs
else:
    n_epochs = 25

if args.h1:
    h1 = args.h1
else:
    h1 = 32

if args.h2:
    h2 = args.h2
else:
    h2 = 64

if args.out:
    out = args.out
else:
    out = 128

if args.learning:
    lr = args.learning
else:
    lr = 0.001

# ----------- Logger and directory set up ------ #
cwd = os.getcwd()
try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project")
    cwd = os.getcwd()
except: # for cluster run
    cwd = os.getcwd()

checkpoint_dir = cwd + '/siamese_gcn/runs/'


# create logger
global logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger('my_log')
# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
time.time()
t = time.strftime('%d%b%y_%H%M%S')
LOG_FILENAME= cwd + '/siamese_gcn/runs/' + t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

logger.info("BATCH SIZE: {}".format(batch_size))
logger.info("NUMBER OF EPOCHS: {}".format(n_epochs))
logger.info("LEARNING RATE: {}".format(lr))
logger.info("H1: {}".format(h1))
logger.info("H2: {}".format(h2))
logger.info("OUT: {}".format(out))

# ------------------------ Load data --------------------- #
X = np.load(cwd+'/std.npy')
Y = np.load(cwd+'/./y.npy')

#X = normalize(X) #does not change anything
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)
n_obs, _ = np.shape(X_train)
print(n_obs)

# ---------------- creating batches -------------------- #
train = ToTorchDataset(X_train, Y_train)
val = ToTorchDataset(X_val, Y_val)
# Creating the batches (balanced classes)
weight = (3/2)*np.ones(len(X_train))
weight[[y==1 for y in Y_train]] = 3
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(X_train), replacement=True)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler = sampler, num_workers=4)
valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)


# --------------------- Training ----------------------- #
# Initialize the network
gcn = GraphConvNetwork(90, h1, h2, out).to(device)
print(gcn.state_dict().keys())
# Define loss and optimizer
criterion = nn.CrossEntropyLoss() # applies softmax + cross entropy
optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=5e-4)
#optimizer = optim.Adadelta(gcn.parameters())
losses = loss_val = acc_val = roc_val = [] 
current_batch_loss = 0
#pos = 0
#neg = 0
# Training loop
for epoch in range(n_epochs): 
    for iter, data in enumerate(trainloader, 0):
        # get the inputs
        labels = data['Y']
        coh_array1 = data['f1']
        coh_array2 = data['f2']
        coh_array3 = data['f3']
        coh_array4 = data['f4']
        coh_array5 = data['f5']
        #coh_array, labels = coh_array.to(device), labels.to(device)
        n, m = coh_array1.size()
        A1 = torch.zeros((n, 90, 90)).to(device)
        A2 = torch.zeros((n, 90, 90)).to(device)
        A3 = torch.zeros((n, 90, 90)).to(device)
        A4 = torch.zeros((n, 90, 90)).to(device)
        A5 = torch.zeros((n, 90, 90)).to(device)
        # we don't have feature so use identity for each graph
        X = torch.eye(90).expand(n, 90, 90)
        for i in range(n):
            A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
            A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
            A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
            A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
            A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))
            #print(A)     
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = gcn(X, A1, A2, A3, A4, A5)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        current_batch_loss += loss.item()

        # to check proprotion of class one
        #pos = np.sum([y==1 for y in labels])
        #neg = np.sum([y==0 for y in labels])
        #print(pos/(pos+neg))

        # to see the gradients
        #params = list(gcn.parameters())
        #for p in params:
        #    print(p.grad)

        #print statistics
        #if iter % 19 == 0:    # print every 100 mini-batches
        #   logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, iter + 1, loss))
    logger.info("Mean of the loss for epoch %d is %.3f" % (epoch + 1, current_batch_loss/(iter+1)))
    current_batch_loss = 0
    loss_val_e, roc_e, acc_e = my_eval(gcn, epoch, i, valloader, batch_size, device, criterion, logger)
    loss_val.append(loss_val_e)
    roc_val.append(roc_e)
    acc_val.append(acc_e)
    torch.save(gcn, checkpoint_dir + t +'.pt')
    losses = list(map(str, losses))
    loss_val = list(map(str, loss_val))
    roc_val = list(map(str, roc_val))
    acc_val = list(map(str, acc_val))
    # save the losses for plotting and monitor training
    with open(checkpoint_dir + t + '_losses.csv', 'w') as outfile:
        outfile.write("\n".join(losses))
    with open(checkpoint_dir + t + '_lossval.csv', 'w') as outfile:
        outfile.write("\n".join(loss_val))
    with open(checkpoint_dir + t + '_rocval.csv', 'w') as outfile:
        outfile.write("\n".join(roc_val))
    with open(checkpoint_dir + t + '_accval.csv', 'w') as outfile:
        outfile.write("\n".join(acc_val))