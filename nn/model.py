import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvNet(nn.Module):
    """ builds the graph conv net for graph classification. 
    Following what is explained in https://github.com/tkipf/gcn/issues/4
    and using the architecture from graph saliency for sex classification.
    """
    def __init__(self, h1=32, h2=32, h3=64, h4=64, h5=128):
        super(GraphConvNet, self).__init__()
        self.h5 = h5
        self.conv1 = nn.Linear(90, h1, bias=False) # input -> hidden 1
        self.conv2 = nn.Linear(h1, h2, bias=False) 
        self.conv3 = nn.Linear(h2, h3, bias=False) 
        self.conv4 = nn.Linear(h3, h4, bias=False) 
        self.conv5 = nn.Linear(h4, h5, bias=False) 
        self.dropout = nn.Dropout(p=0.5, ) #not used b/c does not even converge so no regularization
        self.out = nn.Linear(h5, 2) # graph rep -> output logits

    def forward(self, A_hat, X):
        # get batch size
        b, _, _ = np.shape(A_hat)
        # Graph Conv 
        x = F.relu(self.conv1(A_hat.bmm(X))) # input -> hidden 1
        x = F.relu(self.conv2(A_hat.bmm(x))) # hidden 1 -> node 
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.conv3(A_hat.bmm(x)))
        x = F.relu(self.conv4(A_hat.bmm(x)))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.conv5(A_hat.bmm(x))) 
        x = F.dropout(x, 0.5, training=self.training)
        x = torch.sum(x, 1)
        # output layer (logits)
        x = self.out(x)
        return F.log_softmax(x)