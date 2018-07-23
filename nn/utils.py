import torch
from scipy.linalg import block_diag
from sklearn.metrics import roc_auc_score
import numpy as np


def build_onegraph_A(one_coh_arr, super=False):
    """ builds the A hat matrix of the paper for one
    sample.
    """
    # First construct weighted adjacency matrix
    A = np.zeros((90,90))
    index = np.tril_indices(90)
    A[index]=one_coh_arr
    A = (A + A.T) 
    if super:
        A = np.concatenate((A, np.ones((90, 1))), axis = 1) # adding the super node
        A = np.concatenate((A, np.ones((1, 91))), axis = 0)
    # A tilde from the paper
    di = np.diag_indices(90)
    A[di] = A[di]/2
    A_tilde = A + np.eye(90)
    # D tilde power -0.5
    D_tilde_inv = np.diag(np.power(np.sum(A_tilde, axis=0), -0.5))
    # Finally build A_hat
    A_hat = np.matmul(D_tilde_inv, np.matmul(A_tilde, D_tilde_inv))
    return(A_hat)

def compute_degree(one_coh_arr):
    """ builds the A hat matrix of the paper for one
    sample.
    """
    A_tilde = np.zeros((90,90))
    index = np.tril_indices(90)
    A_tilde[index]=one_coh_arr
    A_tilde = (A_tilde + A_tilde.T)
    deg = np.sum(A_tilde, axis=0)
    return(deg)


def my_eval(gcn, epoch, i, X_valloader, Y_valloader, batch_size, device, criterion, logger):
    """ used for eval on validation set during training.
    """
    #for p in zip(gcn.parameters()):
     #   print('===========\ngradient:\n----------\n{}'.format(p[0].grad))
    with torch.no_grad():
        correct = 0
        total = 0
        proba = list([])
        loss_val = 0
        yvalidation = []
        c = 0
        for xval, yval in zip(X_valloader, Y_valloader):
            # get the inputs
            coh_array, labels = xval.to(device), yval.to(device)
            n, _ = coh_array.size()
            A = torch.zeros((n, 90, 90)).to(device)
            X = torch.zeros((n, 90, 90)).to(device)
            for i in range(n):
                A[i] = torch.tensor(build_onegraph_A(coh_array[i]))
                # we don't have feature so use identity for each graph
                X[i] = torch.eye(90)
            yvalidation.append(labels)
            outputs_val = gcn(A, X)
            proba.append(outputs_val.data.cpu().numpy())
            _, predicted = torch.max(outputs_val.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_val += criterion(outputs_val, labels)
            c += 1
        logger.info('Val loss step [%d, %5d] is: %.3f'%(epoch + 1, i + 1, loss_val/c))
        logger.info('Accuracy of the network val set end of epoch %d : %.3f%% \n and ROC is %.3f' % (epoch + 1, \
                                                        100 * correct / total, \
                                                        roc_auc_score(np.concatenate(yvalidation), np.concatenate(proba)[:,1])))