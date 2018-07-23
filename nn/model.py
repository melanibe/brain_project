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
        # GCN learns node representation
        x = F.relu(self.gc1(feats, adj)) # -> [batch, 90, h1]
        #print(x)
        x = F.relu(self.gc2(x, adj)) # -> [batch, 90, h2]
        x = self.gc3(x, adj) # -> [batch, 90, 128] node representation 
        #print(x)
        #print(x.size())
        
        # Adding GAP layer
        x = torch.sum(x,1) # -> [batch, 128] graph representation summing all nodes
        #print(x.size())
        #print(x)

        # GAP to output layer (logits)
        x = self.out(x) # -> [batch, 2] class of the graph
        return(x)