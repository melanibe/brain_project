import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import build_A_hat

""" what can still be improved
use attention layer (other pooling method) instead of sum the node rep
do some hyperparameter tuning for the layers, paper suggests that is it not necessary to go 
very deep.
"""
class GraphConvNet(nn.Module):
    """builds the graph conv net for graph classification. 
    Following what is explained in https://github.com/tkipf/gcn/issues/4
    """
    def __init__(self, batch_size): # see how nn parameter functions
        super(GraphConvNet, self).__init__()
        #self.device = torch.device(device_name)
        self.batch_size = batch_size
        self.conv1 = nn.Linear(self.batch_size*90, 32) # input -> hidden 1
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Linear(32, 8) # hidden1 -> hidden 2
        self.out = nn.Linear(8, 2) # graph rep -> output logits
        self.sum_pool = np.zeros((self.batch_size,self.batch_size*90))
        for i in range(self.batch_size):
            self.sum_pool[i*self.batch_size:(i+1)*self.batch_size]=1
        self.sum_pool = torch.tensor(self.sum_pool, dtype=torch.float32)
        self.sum_pool = Variable(self.sum_pool, requires_grad=False) # non trainable
        if torch.cuda.is_available():
            self.sum_pool.cuda()

    def forward(self, coh_array):
        coh_array = coh_array.float()
        b, _ = np.shape(coh_array)
        A_hat = build_A_hat(coh_array)
        A_hat = torch.tensor(A_hat, dtype=torch.float32)
        X = torch.eye(90*b)
        if torch.cuda.is_available():
            A_hat.cuda()
            X.cuda()
        A_hat = Variable(A_hat, requires_grad=False) # non trainable
        X = Variable(X, requires_grad=False) # inital node features matrix 90*b
        # GCN(A_hat,  X)
        x = A_hat.mm(X)
        x = self.conv1(x)
        x = F.relu(x) # input -> hidden 1
        x = self.dropout1(x) # drop out
        x = A_hat.mm(x)
        x = F.relu(self.conv2(x)) # hidden 1 -> node representation (hidden 2) output size is [2880, 8]
        # Pooling layer node rep -> graph rep
        x = self.sum_pool.mm(x) # retrieves the graph embedding of dimension 8
        logits = self.out(x) # dense layer
        return logits