import numpy as np
import torch

from model import GraphConvNetwork, GraphConvNetwork_paper, GCN_multiple
from train_utils import training_loop, training_step, val_step
from data_utils import ToTorchDataset, build_onegraph_A, data_to_matrices

from sklearn.base import BaseEstimator, ClassifierMixin

class GCN_estimator_wrapper(BaseEstimator, ClassifierMixin):
    """ this class wraps also the other methods into a estimator class
    thanks to that i can use my gcn as any sklearn estimator
    """
    def __init__(self, checkpoint_file, logger, \
                h1=None, h2=None, out=None, in_feat=90, \
                model_type='normal', batch_size=32, lr = 0.001, n_epochs=20, \
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """ init the model with all the model and training parameters
        Args:
            TO DO
        """
        if model_type=='paper':
            self.gcn = GraphConvNetwork_paper().to(device)
        elif model_type == 'multi':
            self.gcn = GCN_multiple().to(device)
        else:
            self.gcn = GraphConvNetwork(90, h1, h2, out).to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.checkpoint_file = checkpoint_file
        self.logger = logger
    
    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        """ fit = training loop """
        training_loop(self.gcn, X_train, Y_train, \
                        self.batch_size, self.lr, self.n_epochs, \
                        self.device, self.checkpoint_file, self.logger, \
                        X_val, Y_val)
    
    def predict(self, X_test):
        """ predict labels """
        self.gcn.eval()
        test = ToTorchDataset(X_test, None)
        testloader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        y_pred = []
        with torch.no_grad():
            for data in testloader:
                X, A1, A2, A3, A4, A5 = data_to_matrices(data)
                outputs = self.gcn(X, A1, A2, A3, A4, A5)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted.cpu().numpy())
        return(np.asarray(np.concatenate(y_pred)))

    def predict_proba(self, X_test):
        """ predict proba """
        self.gcn.eval()
        test = ToTorchDataset(X_test, None)
        testloader = torch.utils.data.DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        proba = []
        with torch.no_grad():
            for data in testloader:
                X, A1, A2, A3, A4, A5 = data_to_matrices(data)
                outputs = self.gcn(X, A1, A2, A3, A4, A5)
                proba.append(outputs.data.cpu().numpy())
        return(np.asarray(np.concatenate(proba, 0)))