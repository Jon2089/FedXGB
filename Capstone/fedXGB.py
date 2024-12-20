from customxgboost import XGBoostTree, XGBoostClassifier
import numpy as np
import pandas as pd



class Client:
    def __init__(self, X, y):
        self.client_X = X
        self.client_y = y
        self.hist_data = {}
        self.estimators = []
        self.base_y = None
        self.learning_rate = None


    def compute_histograms(self, max_bins=256, global_bins=None):
        X = self.client_X
        y = self.client_y.to_numpy() if isinstance(self.client_y, pd.Series) else self.client_y
        
        if self.estimators:
            y_preds = self.predict(self.client_X)
        elif self.base_y is not None:
            y_preds = np.full((self.client_X.shape[0], 1), self.base_y).flatten().astype('float64')
        else:
            raise ValueError("No initial predictions available.")

        for column in X.columns:
            # Compute quantile-based bins
            if global_bins:
                bin_edges = global_bins[column]
            else:
                bin_edges = np.quantile(X[column], q=np.linspace(0, 1, max_bins + 1))
            
            # Digitize feature values into bins
            bin_indices = np.digitize(X[column], bins=bin_edges, right=False) - 1
            bin_indices[bin_indices == max_bins] = max_bins - 1  # Handle edge case for right boundary
            
            # Preallocate arrays for gradients and Hessians
            gradients = np.zeros(max_bins)
            hessians = np.zeros(max_bins)
            
            # Aggregate gradients and Hessians for each bin
            for bin_idx in range(max_bins):
                indices = np.where(bin_indices == bin_idx)[0]
                if len(indices) > 0:
                    gradients[bin_idx] = np.sum(self.grad(y_preds[indices], y[indices]))
                    hessians[bin_idx] = np.sum(self.hess(y_preds[indices], y[indices]))
            
            # Store results in the histogram data
            self.hist_data[column] = {
                'gradient': gradients,
                'hessian': hessians,
                'bin_edges': bin_edges,
                'values': np.bincount(bin_indices, minlength=max_bins)
            }

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
        self.global_bins = None
        self.global_hist = None
        self.global_y = np.array([])

        self.initialize()

    def initialize(self):
        ''' initialize base preds and global bins '''


        for client in self.clients:
            self.global_y = np.concatenate([self.global_y, client.client_y])

        self.base_predict()

        for client in self.clients:
            client.base_y = self.base_y
            client.compute_histograms(max_bins=self.max_bins)

        # get global bins for each feature, accounting for intersecting bin edges
        global_bins = {}
        for client in self.clients:
            for feature, data in client.hist_data.items():
                if feature not in global_bins:
                    global_bins[feature] = data['bin_edges']
                else:
                    global_bins[feature] = np.unique(np.concatenate([global_bins[feature], data['bin_edges']]))

        # create global histogram
        global_hist = {}
        for feature, bins in global_bins.items():
            global_hist[feature] = {
                'gradient': np.zeros(len(bins) - 1),
                'hessian': np.zeros(len(bins) - 1),
                'bin_edges': bins,
                'values': np.zeros(len(bins) - 1)
            }

        # aggregate gradients and Hessians, if global bins are smaller than client bins, fractionally distribute the gradients and Hessians, otherwise, sum them
        for client in self.clients:
            for feature, data in client.hist_data.items():
                client_bins = data['bin_edges']
                client_gradients = data['gradient']
                client_hessians = data['hessian']

                global_gradients = global_hist[feature]['gradient']
                global_hessians = global_hist[feature]['hessian']
                global_bins = global_hist[feature]['bin_edges']

                # Step 3.1: Aggregate gradients and Hessians (client bins > global bins)
                for i in range(len(global_bins) - 1):
                    # Loop through global bins and aggregate from client bins
                    global_bin_start = global_bins[i]
                    global_bin_end = global_bins[i + 1]

                    # Find the client bins that overlap with the global bin
                    for j in range(len(client_bins) - 1):
                        client_bin_start = client_bins[j]
                        client_bin_end = client_bins[j + 1]

                        # If the global bin overlaps with the client bin
                        if (global_bin_start >= client_bin_start and global_bin_start < client_bin_end) or \
                        (global_bin_end > client_bin_start and global_bin_end <= client_bin_end):
                            # Compute the overlapping length of the bins
                            overlap_start = max(global_bin_start, client_bin_start)
                            overlap_end = min(global_bin_end, client_bin_end)

                            overlap_length = overlap_end - overlap_start
                            client_bin_length = client_bin_end - client_bin_start
                            proportion = overlap_length / client_bin_length

                            # Calculate the contribution of the client bin to the global bin
                            bin_idx = i  # Global bin index
                            global_gradients[bin_idx] += client_gradients[j] * proportion
                            global_hessians[bin_idx] += client_hessians[j] * proportion

        # reduce the global bins to max_bins by aggregation according to quantile method
        for feature, data in global_hist.items():
            global_bins = data['bin_edges']
            global_gradients = data['gradient']
            global_hessians = data['hessian']

            # Reduce the number of bins to max_bins
            bin_edges = np.quantile(global_bins, q=np.linspace(0, 1, self.max_bins + 1))
            new_gradients = np.zeros(self.max_bins)
            new_hessians = np.zeros(self.max_bins)

            for i in range(self.max_bins):
                indices = np.where((global_bins >= bin_edges[i]) & (global_bins < bin_edges[i + 1]))[0]
                if len(indices) > 0:
                    new_gradients[i] = np.sum(global_gradients[indices])
                    new_hessians[i] = np.sum(global_hessians[indices])

            global_hist[feature] = {
                'gradient': new_gradients,
                'hessian': new_hessians,
                'bin_edges': bin_edges
            }



        # store global bins and histogram
        self.global_bins = {feature: data['bin_edges'] for feature, data in global_hist.items()}
        self.global_hist = global_hist


    def histogram_aggregation(self, initial_preds=None):
        """
        Aggregate the histograms from all clients into a global histogram.
        """

        for client in self.clients:
            client.compute_histograms(max_bins=self.max_bins, global_bins=self.global_bins)

        
        # aggregate simply, since global bins are already defined and same for all clients
        for client in self.clients:
            for feature, data in client.hist_data.items():
                client_gradients = data['gradient']
                client_hessians = data['hessian']

                global_gradients = self.global_hist[feature]['gradient']
                global_hessians = self.global_hist[feature]['hessian']

                global_gradients += client_gradients
                global_hessians += client_hessians


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

        self.max_bin = 256
        for booster in range(self.boosting_rounds):

            boosting_tree = XGBoostTree().hist_fit(self.global_hist, subsample_cols = self.subsample_cols, 
                                                   min_leaf = self.min_leaf, min_child_weight = self.min_child_weight, depth = self.depth, 
                                                   lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps)
            
            self.estimators.append(boosting_tree)

            for client in self.clients:
                client.estimators.append(boosting_tree)

            self.histogram_aggregation()

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
    