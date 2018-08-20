import numpy as np 
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import  Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


""" TO DO
- tune layers, number hidden neurons, dropout
- write the model code better (i.e. not copy paste the GCN model but write a separate class for the 
whole model and one for the GCN part)
"""

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
        b, _, _ = np.shape(input)
        hidden = torch.bmm(input, self.weight.expand(b, self.in_features, self.out_features))
        output = torch.bmm(adj, hidden)
        return output

class MySingleGCN(nn.Module):
    def __init__(self, in_feats=90, h1=32, h2=64, out_feats=256):
        super(MySingleGCN, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h1)
        self.gc2 = GraphConvLayer(h1, h2)
        self.gc3 = GraphConvLayer(h2, out_feats)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, feats, adj):
        # GCN learns node representation
        x = F.relu(self.gc1(feats, adj)) # -> [batch, 90, h1]
        x = F.relu(self.gc2(x, adj))
        x = self.dropout(x)
        x = F.relu(self.gc3(x, adj)) # -> [batch, 90, h2]
        return(x)

class GraphConvNetwork(nn.Module):
    def __init__(self, in_feats=90, h1=64, h2=64, out_feats=3):
        super(GraphConvNetwork, self).__init__()
        self.gcn_node = MySingleGCN(in_feats, h1, h2, out_feats)
        self.out = nn.Linear(5*out_feats, 2)
    
    def forward(self, feats, adj1, adj2, adj3, adj4, adj5):
        # GCN learns node representation - same weights for all
        x1 = self.gcn_node(feats, adj1)
        x2 = self.gcn_node(feats, adj2)
        x3 = self.gcn_node(feats, adj3)
        x4 = self.gcn_node(feats, adj4)
        x5 = self.gcn_node(feats, adj5)
       
        # Adding GAP layer
        x1 = torch.sum(x1,1) # -> [batch, 128] graph representation summing all nodes
        x2 = torch.sum(x2,1)
        x3 = torch.sum(x3,1)
        x4 = torch.sum(x4,1)
        x5 = torch.sum(x5,1)

        # concatenating features from all graphs
        x = torch.cat((x1,x2,x3,x4,x5), 1)

        # GAP to output layer (logits)
        x = self.out(x) # -> [batch, 2] class of the graph
        return(x)








