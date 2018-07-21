import torch
from scipy.linalg import block_diag
from sklearn.metrics import roc_auc_score
import numpy as np


def build_onegraph_A(one_coh_arr):
    """ builds the A hat matrix of the paper for one
    sample.
    """
    A_tilde = np.zeros((90,90))
    index = np.tril_indices(90)
    A_tilde[index]=one_coh_arr
    A_tilde = (A_tilde + A_tilde.T)
    di = np.diag_indices(90)
    A_tilde[di] = 1
    D_tilde_inv = np.diag(1/np.sqrt(np.sum(A_tilde, axis=0)))
    D_tilde = np.diag(np.sqrt(np.sum(A_tilde, axis=0)))
    A_hat = np.matmul(D_tilde_inv, np.matmul(A_tilde, D_tilde))
    return(A_hat)

def build_A_hat(X_array):
    """ builds the block diagonal A hat matrix
    """
    batch_size, _ = np.shape(X_array)
    list_blocks =[build_onegraph_A(X_array[i]) for i in range(batch_size)]
    A_hat = block_diag(*list_blocks)
    return(A_hat)


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
        for xval, yval in zip(X_valloader, Y_valloader):
            if (len(yval)==batch_size):
                yvalidation.append(yval)
                xval, yval = xval.to(device), yval.to(device)
                outputs_val = gcn(xval)
                proba.append(outputs_val.data.cpu().numpy())
                _, predicted = torch.max(outputs_val.data, 1)
                total += yval.size(0)
                correct += (predicted == yval).sum().item()
                loss_val += criterion(outputs_val, yval)
        logger.info('Val loss step [%d, %5d] is: %.3f'%(epoch + 1, i + 1, loss_val/total))
        logger.info('Accuracy of the network val set end of epoch  %d : %.3f%%' % (epoch + 1, 100 * correct / total))
        logger.info('ROC is %.3f '%(roc_auc_score(np.concatenate(yvalidation), np.concatenate(proba)[:,1])))