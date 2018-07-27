import logging
import time
import os 

import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag

from model import GraphConvNetwork, GraphConvNetwork_paper, GCN_multiple
from data_utils import build_onegraph_A, ToTorchDataset


def training_step(gcn, data, optimizer, criterion, device):
    gcn.train()
    # get the inputs
    labels = data['Y']
    coh_array1 = data['f1']
    coh_array2 = data['f2']
    coh_array3 = data['f3']
    coh_array4 = data['f4']
    coh_array5 = data['f5']

    n, m = coh_array1.size()
    A1 = torch.zeros((n, 90, 90)).to(device)
    A2 = torch.zeros((n, 90, 90)).to(device)
    A3 = torch.zeros((n, 90, 90)).to(device)
    A4 = torch.zeros((n, 90, 90)).to(device)
    A5 = torch.zeros((n, 90, 90)).to(device)

    # we don't have feature so use identity for each graph
    X = torch.eye(90).expand(n, 90, 90)
    for i in range(n):
        A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
        A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
        A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
        A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
        A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))
        #print(A)     
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = gcn(X, A1, A2, A3, A4, A5)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return(loss.item())

def val_step(gcn, valloader, batch_size, device, criterion, logger):
    """ used for eval on validation set during training.
    """
    gcn.eval()
    correct = 0
    total = 0
    proba = list([])
    loss_val = 0
    yvalidation = []
    c = 0
    with torch.no_grad():
        for data in valloader:
            # get the inputs
            labels = data['Y']
            coh_array1 = data['f1']
            coh_array2 = data['f2']
            coh_array3 = data['f3']
            coh_array4 = data['f4']
            coh_array5 = data['f5']
        #coh_array, labels = coh_array.to(device), labels.to(device)
            n, m = coh_array1.size()
            A1 = torch.zeros((n, 90, 90)).to(device)
            A2 = torch.zeros((n, 90, 90)).to(device)
            A3 = torch.zeros((n, 90, 90)).to(device)
            A4 = torch.zeros((n, 90, 90)).to(device)
            A5 = torch.zeros((n, 90, 90)).to(device)
            X = torch.eye(90).expand(n, 90, 90)
            for i in range(n):
                A1[i] = torch.tensor(build_onegraph_A(coh_array1[i]))
                A2[i] = torch.tensor(build_onegraph_A(coh_array2[i]))
                A3[i] = torch.tensor(build_onegraph_A(coh_array3[i]))
                A4[i] = torch.tensor(build_onegraph_A(coh_array4[i]))
                A5[i] = torch.tensor(build_onegraph_A(coh_array5[i]))
            yvalidation.append(labels)
            outputs_val = gcn(X, A1, A2, A3, A4, A5)
            proba.append(outputs_val.data.cpu().numpy())
            _, predicted = torch.max(outputs_val.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_val += criterion(outputs_val, labels)
            c += 1
        roc = roc_auc_score(np.concatenate(yvalidation), np.concatenate(proba)[:,1])
        acc = correct / total
        logger.info('Val loss is: %.3f'%(loss_val/c))
        logger.info('Accuracy of the network val set : %.3f%% \n and ROC is %.3f' % (100*acc, roc))
    return(loss_val/c, roc, acc)

def training_loop(gcn, X_train, Y_train, batch_size, lr, n_epochs, device, checkpoint_file, logger, X_val=None, Y_val=None):
    train = ToTorchDataset(X_train, Y_train)
    if X_val is not None:
        val = ToTorchDataset(X_val, Y_val)
        valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)
    # Creating the batches (balanced classes)
    weight = (3/2)*np.ones(len(X_train))
    weight[[y==1 for y in Y_train]] = 3
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(X_train), replacement=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler = sampler, num_workers=4)
    # Define loss and optimizer
    #weight = torch.tensor([1,2], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss() # applies softmax + cross entropy
    optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=5e-4)
    losses = loss_val = acc_val = roc_val = [] 
    current_batch_loss = 0
    # Training loop
    for epoch in range(n_epochs): 
        for iter, data in enumerate(trainloader, 0):
            current_loss = training_step(gcn, data, optimizer, criterion, device)
            current_batch_loss += current_loss
            losses += [current_loss]
        #print statistics in the epoch might be better - to do
        logger.info("Mean of the loss for epoch %d is %.3f" % (epoch + 1, current_batch_loss/(iter+1)))
        current_batch_loss = 0
        if X_val is not None:
            loss_val_e, roc_e, acc_e = val_step(gcn, valloader, batch_size, device, criterion, logger)
            loss_val.append(loss_val_e)
            roc_val.append(roc_e)
            acc_val.append(acc_e)
    torch.save(gcn, checkpoint_file +'.pt')
    losses = list(map(str, losses))
    with open(checkpoint_file + '_losses.csv', 'w') as outfile:
        outfile.write("\n".join(losses))
    if X_val is not None:
        loss_val = list(map(str, loss_val))
        roc_val = list(map(str, roc_val))
        acc_val = list(map(str, acc_val))
        # save the losses for plotting and monitor training
        with open(checkpoint_file + '_lossval.csv', 'w') as outfile:
            outfile.write("\n".join(loss_val))
        with open(checkpoint_file + '_rocval.csv', 'w') as outfile:
            outfile.write("\n".join(roc_val))
        with open(checkpoint_file + '_accval.csv', 'w') as outfile:
            outfile.write("\n".join(acc_val))



