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
from utils import my_eval, build_onegraph_A
from scipy.linalg import block_diag
# ----------- To allow run on GPU ------------ #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------------ Global parameters ------------- #

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", help="batch_size", type=int)
parser.add_argument("-e", "--epochs", help="number of epochs", type=int)
parser.add_argument("-h1", "--h1", help="size hidden 1", type=int)
parser.add_argument("-h2", "--h2", help="size hidden 2", type=int)
parser.add_argument("-lr", "--learning", help="learning rate", type=float)
args = parser.parse_args()

if args.batch:
    batch_size = args.batch
else:
    batch_size = 32

if args.epochs:
    n_epochs = args.epochs
else:
    n_epochs = 200

if args.h1:
    h1 = args.h1
else:
    h1 = 32

if args.h2:
    h2 = args.h2
else:
    h2 = 128

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
#logger.info("H1: {}".format(h1))
#logger.info("H2: {}".format(h2))
logger.info("learning rate: {}".format(lr))

# ------------------------ Load data --------------------- #
X = np.load(cwd+'/one.npy')
Y = np.load(cwd+'/./y.npy')
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)
n_obs, _ = np.shape(X_train)
# Creating the batches (balanced classes)
weight = (3/2)*np.ones(len(X_train))
weight[Y_main==1] = 3
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
X_trainloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, sampler = sampler, num_workers=4)
Y_trainloader = torch.utils.data.DataLoader(Y_train, batch_size=batch_size, sampler = sampler, num_workers=4)
X_valloader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False, num_workers=4)
Y_valloader = torch.utils.data.DataLoader(Y_val, batch_size=batch_size, shuffle=False, num_workers=4)



# --------------------- Training ----------------------- #

# Initialize the network
gcn = GraphConvNet().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=5e-4)
#optimizer = optim.Adadelta(gcn.parameters())
losses = [] 
current_batch_loss = 0
# Training loop
for epoch in range(n_epochs): 
    for i, data in enumerate(zip(X_trainloader, Y_trainloader), 0):
        # get the inputs
        coh_array, labels = data
        coh_array, labels = coh_array.to(device), labels.to(device)
        n, _ = coh_array.size()
        A = torch.zeros((n, 90, 90)).to(device)
        X = torch.zeros((n, 90, 90)).to(device)
        for i in range(n):
            A[i] = torch.tensor(build_onegraph_A(coh_array[i]))
            # we don't have feature so use identity for each graph
            X[i] = torch.eye(90)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = gcn(A, X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses += [str(loss.item())]
        current_batch_loss += loss.item()
        #print statistics
        if i % 19 == 0:    # print every 100 mini-batches
            logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
    logger.info("Mean of the loss for epoch %d is %.3f" % (epoch + 1, current_batch_loss/(i+1)))
    current_batch_loss = 0
    if epoch%10 == 0:
        my_eval(gcn, epoch, i, X_valloader, Y_valloader, batch_size, device, criterion, logger)
    #    torch.save(gcn, checkpoint_dir + t +'.pt')
    # save the losses for plotting and monitor training
    #with open(checkpoint_dir + t + '_losses.csv', 'w') as outfile:
    #    outfile.write("\n".join(losses))
