from customxgboost import XGBoostTree, XGBoostClassifier
import numpy as np
import pandas as pd
from histogram import Histogram



class Client:
    def __init__(self, X, y, max_bins=256):
        self.client_X = X
        self.client_y = y
        self.estimators = []
        self.base_y = None
        self.learning_rate = None
        self.max_bins = max_bins
        self.histogram = Histogram(max_bins=256, client_initiate=True)
        self.feature_splits = self.histogram.feature_splits


    def compute_histograms(self, global_feature_splits):
        X = self.client_X
        y = self.client_y.to_numpy() if isinstance(self.client_y, pd.Series) else self.client_y
        
        if self.estimators:
            y_preds = self.predict(self.client_X)
        elif self.base_y is not None:
            y_preds = np.full((self.client_X.shape[0], 1), self.base_y).flatten().astype('float64')
        else:
            raise ValueError("No initial predictions available.")
        
        self.histogram = Histogram(feature_splits=global_feature_splits)
        Grads = self.grad(y_preds, self.client_y)
        Hess = self.hess(y_preds, self.client_y)
        self.histogram.compute_histogram(self.client_X)
        
        return self.histogram.histogram
        

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # first order gradient logLoss
    def grad(self, preds, labels):
        preds = self.sigmoid(preds)
        return(preds - labels)
    
    # second order gradient logLoss
    def hess(self, preds, labels):
        preds = self.sigmoid(preds)
        return(preds * (1 - preds))
    
    def predict(self, X):
        # X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        pred = np.full((X.shape[0], 1), self.base_y).flatten().astype('float64')
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
        
        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return(preds)










class FedXGBoost(XGBoostClassifier):
    
    def __init__(self, clients, max_bins=256):
        super().__init__()
        self.clients = clients
        self.max_bins = max_bins
        self.global_feature_splits = []
        self.global_y = np.array([])

        self.initialize()

    def initialize(self):
        ''' initialize base preds and global bins '''
        
        for client in self.clients:
            self.global_y = np.concatenate([self.global_y, client.client_y])
            self.global_feature_splits += client.feature_splits

        self.base_predict()

        self.global_feature_splits = np.quantile(self.global_feature_splits, q=np.linspace(0, 1, self.max_bins + 1))

        for client in self.clients:
            client.base_y = self.base_y


    def histogram_aggregation(self, initial_preds=None):
        """
        Aggregate the histograms from all clients into a global histogram.
        """

        self.global_histogram = None
        for client in self.clients:
            client_histogram = client.compute_histogram(self.global_feature_splits)
            if self.global_histogram is None:
                self.global_histogram = client_histogram
            
            self.global_histogram['gradients'] += client_histogram['gradients']
            self.global_histogram['hessians'] += client_histogram['hessians']
            self.global_histogram['counts'] += client_histogram['counts']


    def base_predict(self):
        # calclate binary classification base_pred
        P = np.mean(self.global_y)
        base_pred = np.log(P / (1 - P))
        self.base_y = base_pred 


    
    def fit(self, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):

        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma

        for client in self.clients:
            client.learning_rate = self.learning_rate

        for booster in range(self.boosting_rounds):
            self.histogram_aggregation()
            boosting_tree = XGBoostTree().hist_fit(self.global_histogram, max_bins=self.max_bins, feature_splits=self.global_feature_splits, subsample_cols = self.subsample_cols, 
                                                   min_leaf = self.min_leaf, min_child_weight = self.min_child_weight, depth = self.depth, 
                                                   lambda_ = self.lambda_, gamma = self.gamma)
            
            self.estimators.append(boosting_tree)

            for client in self.clients:
                client.estimators.append(boosting_tree)


        print('Training Complete')


    def predict_proba(self, X):
        # pred is a 1D array of all values equal to self.base_y of shape (X.shape[0],): [y, y, y, y ...]
        np.full((X.shape[0], 1), self.base_y).flatten().astype('float64')

        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))
    
    def predict(self, X):
        # X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        pred = np.full((X.shape[0], 1), self.base_y).flatten().astype('float64')
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
        
        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return(preds)
    