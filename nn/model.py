import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import  Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import numpy as np 

import math

class GraphConvLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        input(torch.Tensor):
        adj(torch.Tensor): adjacency matrix
        '''
        b, _, _ = np.shape(input)
        hidden = torch.bmm(input, self.weight.expand(b, self.in_features, self.out_features))
        output = torch.bmm(adj, hidden)
        return output

class GraphConvNetwork(nn.Module):
    def __init__(self, in_feats=90, hidden_size=64, out_feats=128):
        super(GraphConvNetwork, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.gc3 = GraphConvLayer(hidden_size, out_feats)
        self.out = nn.Linear(out_feats, 2)
    
    def forward(self, feats, adj):
        '''
        feats(torch.tensor): the graph node features(matrix of Nxd), N = num nodes, d = num features
        adj(torch.tensor): adjacency matrix of the graph
        '''
        #print(adj)
        x = self.gc1(feats, adj)
        #print(x)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        #print(x)
        #print(x.size())
        x = torch.sum(x,1)
        #print(x.size())
       # print(x)
        # output layer (logits)
        x = self.out(x)
        return(x)