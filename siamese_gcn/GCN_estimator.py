import numpy as np
import torch

from siamese_gcn.model import GraphConvNetwork
from siamese_gcn.train_utils import training_loop, training_step, val_step
from siamese_gcn.data_utils import ToTorchDataset, build_onegraph_A, data_to_matrices

from sklearn.base import BaseEstimator, ClassifierMixin

class GCN_estimator_wrapper(BaseEstimator, ClassifierMixin):
    """ Wrapper for the Graph Convoluational network.
    Necessary in order to use this network just as it was 
    any sklearn estimator in the (custom) cross validation.
    """
    def __init__(self, checkpoint_dir, logger, \
                h1=None, h2=None, out=None, in_feat=90, \
                batch_size=32, lr=0.001, nsteps=1000, \
                reset = False):
        """ Init the model from GraphConvNetwork object.
        
        Args:
            checkpoint_dir(str): name of the checkpoint directory for the run.
            logger(logger): logger object to print the results to.
            h1: dimension of the first hidden layer
            h2: dimension of the second hidden layer
            out: dimension of the node features
            in_feat: dimension of the input features i.e. number of nodes in the graph
            batch_size: batch_size
            lr(float): learning rate for the optimizer 
            nsteps: number of training steps to apply
            reset(bool): whether to reset the network each time fit is called.
                        Set to true in cross-validation setting.
        """
        self.gcn = GraphConvNetwork(90, h1, h2, out)
        self.batch_size = batch_size
        self.lr = lr
        self.nsteps = nsteps
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.h1 = h1
        self.h2 = h2
        self.out = out
        self.reset = reset # reset the network at each fit call ? TRUE for CV !!!
        logger.info("Success init of GCN params {}-{}-{}".format(self.h1, self.h2, self.out))
        logger.info("Training parameters {} steps and {} learning rate".format(self.nsteps, self.lr))
    
    def fit(self, X_train, Y_train, X_val=None, Y_val=None, filename=""):
        """ fit = training loop """
        if self.reset:
            self.gcn = GraphConvNetwork(90, self.h1, self.h2, self.out)
        training_loop(self.gcn, X_train, Y_train, \
                        self.batch_size, self.lr, \
                        self.logger, self.checkpoint_dir, filename, \
                        X_val, Y_val, nsteps=self.nsteps)
    
    def predict(self, X_test):
        """ predict labels """
        self.gcn.eval()
        test = ToTorchDataset(np.asarray(X_test))
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