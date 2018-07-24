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

class GraphConvNetwork(nn.Module):
    def __init__(self, in_feats=90, h1=64, h2=64, out_feats=3):
        super(GraphConvNetwork, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h1)
        self.gc2 = GraphConvLayer(h1, h2)
        self.gc3 = GraphConvLayer(h2, out_feats)
        self.out = nn.Linear(5*out_feats, 2)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, feats, adj1, adj2, adj3, adj4, adj5):
        # GCN learns node representation
        x1 = F.relu(self.gc1(feats, adj1)) # -> [batch, 90, h1]
        #x1 = self.dropout(x1)
        x1 = F.relu(self.gc2(x1, adj1)) # -> [batch, 90, h2]
        x1 = self.dropout(x1)
        x1 = self.gc3(x1, adj1) # -> [batch, 90, 128] node representation 

        x2 = F.relu(self.gc1(feats, adj2)) # -> [batch, 90, h1]
        #x2 = self.dropout(x2)
        x2 = F.relu(self.gc2(x2, adj2)) # -> [batch, 90, h2]
        x2 = self.dropout(x2)
        x2 = self.gc3(x2, adj2) 

        x3 = F.relu(self.gc1(feats, adj3)) # -> [batch, 90, h1]
        #x3 = self.dropout(x3)
        x3 = F.relu(self.gc2(x3, adj3)) # -> [batch, 90, h2]
        x3 = self.dropout(x3)
        x3 = self.gc3(x3, adj3) 

        x4 = F.relu(self.gc1(feats, adj4)) # -> [batch, 90, h1]
        #x4 = self.dropout(x4)
        x4 = F.relu(self.gc2(x4, adj4)) # -> [batch, 90, h2]
        x4 = self.dropout(x4)
        x4 = self.gc3(x4, adj4)         

        x5 = F.relu(self.gc1(feats, adj5)) # -> [batch, 90, h1]
        #x5 = self.dropout(x5)
        x5 = F.relu(self.gc2(x5, adj5)) # -> [batch, 90, h2]
        x5 = self.dropout(x5)
        x5 = self.gc3(x5, adj5)
       
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