import logging
import time
import os 
import torch
import numpy as np
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import GraphConvNet
from utils import my_eval

# ----------- To allow run on GPU ------------ #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ------------ Global parameters ------------- #
batch_size = 32
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", help="batch_size", type=int)
parser.add_argument("-e", "--epochs", help="number of epochs")
args = parser.parse_args()

if args.batch:
    batch_size = args.batch
else:
    batch_size = 32

if args.epochs:
    n_epochs = args.epochs
else:
    n_epochs = 200

# ----------- Logger and directory set up ------ #
cwd = os.getcwd()
try: # for local run
    os.chdir("/Users/melaniebernhardt/Documents/brain_project")
    cwd = os.getcwd()
except: # for cluster run
    cwd = os.getcwd()

checkpoint_dir = cwd + '/nn/runs/'


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
LOG_FILENAME= cwd + '/nn/runs/' + t +'.log'
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

logger.info("BATCH SIZE: {}".format(batch_size))
logger.info("NUMBER OF EPOCHS: {}".format(n_epochs))
# ------------------------ Load data --------------------- #
X = np.load(cwd+'/./one.npy')
Y = np.load(cwd+'/./y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)

# Creating the batches (balanced classes)
weight = (3/2)*np.ones(len(X_train))
weight[Y_main==1] = 3
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
# further splitting intial train into train and val (don't use test for val)
X_trainloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, sampler = sampler, num_workers=4)
Y_trainloader = torch.utils.data.DataLoader(Y_train, batch_size=batch_size, sampler = sampler, num_workers=4)
X_valloader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False, num_workers=4)
Y_valloader = torch.utils.data.DataLoader(Y_val, batch_size=batch_size, shuffle=False, num_workers=4)

# --------------------- Training ----------------------- #

# Initialize the network
gcn = GraphConvNet(batch_size).to(device)

# Define loss and optimizer
w = torch.tensor([1,1], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=w)
optimizer = optim.Adam(gcn.parameters(), lr=0.000001)
losses = [] 
# Training loop
for epoch in range(n_epochs):  # loop over the dataset multiple times
    for i, data in enumerate(zip(X_trainloader, Y_trainloader), 0):
        # get the inputs
        coh_array, labels = data
        coh_array, labels = coh_array.to(device), labels.to(device)
        if len(labels)==batch_size:
            # zero the parameter gradients
            optimizer.zero_grad() 
            # forward + backward + optimize
            outputs = gcn(coh_array)
            loss = criterion(outputs, labels)
            losses += [str(loss.item())]
            loss.backward()
            optimizer.step()
            # print statistics
            if i % 19 == 0:    # print every 100 mini-batches
                logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
    my_eval(gcn, epoch, i, X_valloader, Y_valloader, batch_size, device, criterion, logger)
    torch.save(gcn, checkpoint_dir + t +'.pt')
    with open(checkpoint_dir + t + '_losses.csv', 'w') as outfile:
        outfile.write("\n".join(losses))
