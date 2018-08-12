import numpy as np 
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import  Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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


class GCN_multiple(nn.Module):
    "same as normal but different node learning for each freq"
    def __init__(self, in_feats=90, h1=32, h2=64, out_feats=256):
        super(GCN_multiple, self).__init__()
        self.gcn1 = MySingleGCN(in_feats, h1, h2, out_feats)
        self.gcn2 = MySingleGCN(in_feats, h1, h2, out_feats)
        self.gcn3 = MySingleGCN(in_feats, h1, h2, out_feats)
        self.gcn4 = MySingleGCN(in_feats, h1, h2, out_feats)
        self.gcn5 = MySingleGCN(in_feats, h1, h2, out_feats)
        self.out = nn.Linear(5*out_feats, 2)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, feats, adj1, adj2, adj3, adj4, adj5):
        # GCN learns node representation
        x1 = self.gcn1(feats, adj1)
        x2 = self.gcn2(feats, adj2)
        x3 = self.gcn3(feats, adj3)
        x4 = self.gcn4(feats, adj4)
        x5 = self.gcn5(feats, adj5)     
       
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



class GCN_10(nn.Module):
    def __init__(self, in_feats=90, h1=128, h2=64, out_feats=128):
        super(GCN_10, self).__init__()
        self.gcn = MySingleGCN(in_feats, h1, h2, out_feats)
        self.out = nn.Linear(10*out_feats, 2)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, feats, adj1, adj2, adj3, adj4, adj5,adj6, adj7, adj8, adj9, adj10):
        # GCN learns node representation
        x1 = self.gcn(feats, adj1)
        x2 = self.gcn(feats, adj2)
        x3 = self.gcn(feats, adj3)
        x4 = self.gcn(feats, adj4)
        x5 = self.gcn(feats, adj5)
        x6 = self.gcn(feats, adj6)
        x7 = self.gcn(feats, adj7)
        x8 = self.gcn(feats, adj8)
        x9 = self.gcn(feats, adj9)
        x10 = self.gcn(feats, adj10)       
       
        # Adding GAP layer
        x1 = torch.sum(x1,1) # -> [batch, 128] graph representation summing all nodes
        x2 = torch.sum(x2,1)
        x3 = torch.sum(x3,1)
        x4 = torch.sum(x4,1)
        x5 = torch.sum(x5,1)
        x6 = torch.sum(x6,1) # -> [batch, 128] graph representation summing all nodes
        x7 = torch.sum(x7,1)
        x8 = torch.sum(x8,1)
        x9 = torch.sum(x9,1)
        x10 = torch.sum(x10,1)

        # concatenating features from all graphs
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), 1)

        # GAP to output layer (logits)
        x = self.out(x) # -> [batch, 2] class of the graph
        return(x)


class GraphConvNetwork_paper(nn.Module):
    "deeper arch in sex classification paper"
    def __init__(self, in_feats=90, h1=128, h2=320, out_feats=128):
        super(GraphConvNetwork_paper, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h1)
        self.gc2 = GraphConvLayer(h1, h1)
        self.gc3 = GraphConvLayer(h1, h2)
        self.gc4 = GraphConvLayer(h2, h2)
        self.gc5 = GraphConvLayer(h2, out_feats)
        self.out = nn.Linear(5*out_feats, 2)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, feats, adj1, adj2, adj3, adj4, adj5):
        # GCN learns node representation
        x1 = F.relu(self.gc1(feats, adj1)) # -> [batch, 90, h1]
        x1 = F.relu(self.gc2(x1, adj1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.gc3(x1, adj1)) # -> [batch, 90, h2]
        x1 = F.relu(self.gc4(x1, adj1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.gc5(x1, adj1))
        x1 = self.dropout(x1)

        x2 = F.relu(self.gc1(feats, adj2)) # -> [batch, 90, h1]
        x2 = F.relu(self.gc2(x2, adj2))
        x2 = self.dropout(x2)
        x2 = F.relu(self.gc3(x2, adj2)) # -> [batch, 90, h2]
        x2 = F.relu(self.gc4(x2, adj2))
        x2 = self.dropout(x2)
        x2 = F.relu(self.gc5(x2, adj2))
        x2 = self.dropout(x2)

        x3 = F.relu(self.gc1(feats, adj3)) # -> [batch, 90, h1]
        x3 = F.relu(self.gc2(x3, adj3))
        x3 = self.dropout(x3)
        x3 = F.relu(self.gc3(x3, adj3)) # -> [batch, 90, h2]
        x3 = F.relu(self.gc4(x3, adj3))
        x3 = self.dropout(x3)
        x3 = F.relu(self.gc5(x3, adj3))
        x3 = self.dropout(x3)

        x4 = F.relu(self.gc1(feats, adj4)) # -> [batch, 90, h1]
        x4 = F.relu(self.gc2(x4, adj4))
        x4 = self.dropout(x4)
        x4 = F.relu(self.gc3(x4, adj4)) # -> [batch, 90, h2]
        x4 = F.relu(self.gc4(x4, adj4))
        x4 = self.dropout(x4)
        x4 = F.relu(self.gc5(x4, adj4))
        x4 = self.dropout(x4)

        x5 = F.relu(self.gc1(feats, adj5)) # -> [batch, 90, h1]
        x5 = F.relu(self.gc2(x5, adj5))
        x5 = self.dropout(x5)
        x5 = F.relu(self.gc3(x5, adj5)) # -> [batch, 90, h2]
        x5 = F.relu(self.gc4(x5, adj5))
        x5 = self.dropout(x5)
        x5 = F.relu(self.gc5(x5, adj5))
        x5 = self.dropout(x5)
       
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




