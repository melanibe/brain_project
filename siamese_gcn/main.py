import logging
import time
import os 
import numpy as np
import argparse
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from model import GraphConvNetwork, GraphConvNetwork_paper, GCN_multiple
from GCN_estimator import GCN_estimator_wrapper

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
parser.add_argument("-m", "--model", help="model")
args = parser.parse_args()

if args.model:
    model = args.model
else:
    model = 'GCN'

if args.batch:
    batch_size = args.batch
else:
    batch_size = 32

if args.epochs:
    n_epochs = args.epochs
else:
    n_epochs = 10

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
if not model=='paper':
    logger.info("H1: {}".format(h1))
    logger.info("H2: {}".format(h2))
    logger.info("OUT: {}".format(out))

checkpoint_file = checkpoint_dir + t

# ------------------------ Load data --------------------- #
X = np.load(cwd+'/matrices/std.npy')
Y = np.load(cwd+'/matrices/y.npy')

#X = normalize(X) #does not change anything
Y_main = [1 if ((y==1) or (y==2)) else 0 for y in Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_main, test_size=0.3, random_state=42, stratify=Y_main)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, stratify=Y_train)
n_obs, _ = np.shape(X_train)
print(n_obs)

# Initialize the network
if model=='paper':
    logger.info("USING PAPER MODEL")
elif model == 'multi':
    logger.info("USING MULTIPLE MODEL")
else:
    logger.info("NORMAL MODEL")

gcn = GCN_estimator_wrapper(checkpoint_file, logger, h1, h2, out, n_epochs=5)
gcn.fit(X_train, Y_train, X_val, Y_val)
pred = gcn.predict(X_val)

print(confusion_matrix(Y_val, pred))
print(accuracy_score(Y_val, pred))

proba = gcn.predict_proba(X_val)
print(roc_auc_score(Y_val, proba[:,1]))
