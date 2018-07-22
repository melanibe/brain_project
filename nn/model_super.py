import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from utils import build_A_hat
from scipy.linalg import block_diag


class GraphConvNet(nn.Module):
    """ builds the graph conv net for graph classification. 
    Following what is explained in https://github.com/tkipf/gcn/issues/4
    """
    def __init__(self, h1, h2, f1=128):
        super(GraphConvNet, self).__init__()
        self.h1 = h1
        self.h2 = h2
        self.conv1 = nn.Linear(91, h1, bias=False) # input -> hidden 1
        #self.dropout1 = nn.Dropout(p=0.01) not used b/c does not even converge so no regularization
        self.conv2 = nn.Linear(h1, h2, bias=False) # hidden1 -> hidden 2
        self.out1 = nn.Linear(h2, f1) # graph rep -> output logits
        self.out2 = nn.Linear(f1, 2) # graph rep -> output logits

    def forward(self, coh_array):
        # original array of pairs
        coh_array = coh_array.float()
        # get batch size
        b, _ = np.shape(coh_array)
        # build A_hat as in the paper
        A_hat = build_A_hat(coh_array, super=True)
        # to tensor
        A_hat = torch.tensor(A_hat, dtype=torch.float32, requires_grad=False) # non trainable
        # we don't have feature so use identity for each graph
        id91 = torch.eye(91, requires_grad=False)
        # concatenate the feature matrix for batch of graphs
        X = torch.cat([id91 for i in range(b)], dim = 0) 
        # build the outpooling matrix (could be done outside in train.py)     
        sum_pool = torch.zeros((b,91*b), requires_grad=False)
        for i in range(b):
            sum_pool[i, i*91+90]=1
        # send the tensors to GPU if necessary
        if torch.cuda.is_available():
            sum_pool = sum_pool.to(torch.device("cuda:0"))
            A_hat = A_hat.to(torch.device("cuda:0"))
            X = X.to(torch.device("cuda:0"))
        # Graph Conv 
        x = F.relu(self.conv1(A_hat.mm(X))) # input -> hidden 1
        x = self.conv2(A_hat.mm(x)) # hidden 1 -> node 
        # pool the node representation from GCN to get graph representation
        x = sum_pool.mm(x)
        # fully connected layer for graph classification
        fc1 = F.relu(self.out1(x))
        # output layer (logits)
        logits = self.out2(fc1)
        return logits