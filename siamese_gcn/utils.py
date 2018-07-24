import torch
from scipy.linalg import block_diag
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.utils.data import Dataset


class ToTorchDataset(Dataset):
    """From matrix to torch dataset follwing tutorial"""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        f1 = torch.from_numpy(self.X[idx, 0:4095])
        f2 = torch.from_numpy(self.X[idx, 4095:2*4095])
        f3 = torch.from_numpy(self.X[idx, 2*4095:3*4095])
        f4 = torch.from_numpy(self.X[idx, 3*4095:4*4095])
        f5 = torch.from_numpy(self.X[idx, 4*4095:5*4095])
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        sample = {'Y': y, 'f1': f1, 'f2':f2, 'f3':f3, 'f4':f4, 'f5':f5}
        return sample


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



def my_eval(gcn, epoch, i, valloader, batch_size, device, criterion, logger):
    """ used for eval on validation set during training.
    """
    with torch.no_grad():
        correct = 0
        total = 0
        proba = list([])
        loss_val = 0
        yvalidation = []
        c = 0
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
        logger.info('Val loss step [%d, %5d] is: %.3f'%(epoch + 1, i + 1, loss_val/c))
        logger.info('Accuracy of the network val set end of epoch %d : %.3f%% \n and ROC is %.3f' % (epoch + 1, \
                                                        100*acc, \
                                                        roc))
    return(loss_val/c, roc, acc)

