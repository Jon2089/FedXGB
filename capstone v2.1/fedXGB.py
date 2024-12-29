from customxgboost import XGBoostTree, XGBoostClassifier
import numpy as np
import pandas as pd
from histogram import Histogram
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, as_completed



class Client:
    def __init__(self, X, y, client_id):
        self.id = client_id
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.samples = X.shape[0]
        self.estimators = []
        self.base_y = None
        self.learning_rate = None
        self.n_quantiles = int(np.ceil(self.X.shape[0] / 10) + 2)
        self.get_feature_importance()
        self.get_binary_features()
        self.get_quantiles()

    def compute_histogram(self, feature_splits, compute_regions):
        if self.estimators:
            self.y_preds += self.learning_rate * self.estimators[-1].predict(self.X)
        elif self.base_y is not None:
            self.y_preds = np.full((self.X.shape[0], 1), self.base_y).flatten().astype('float64')
        else:
            raise ValueError("No initial predictions available.")
        
        
        if compute_regions:
            self.histogram = Histogram(feature_splits=feature_splits)
            self.histogram.fit(self.X)
        Grads = self.grad(self.y_preds, self.y)
        Hess = self.hess(self.y_preds, self.y)
        self.histogram.compute_histogram(Grads, Hess)
        
        return self.histogram.histogram
        

    def get_quantiles(self):
        '''
        Returns the quantiles of the features.
        '''

        self.quantiles = {}
        for i in range(self.X.shape[1]):
            if i not in self.binary:
                self.quantiles[i] = np.quantile(self.X[:, i], q=np.linspace(0, 1, self.n_quantiles)).tolist()
            else:
                self.quantiles[i] = list(self.binary[i])


    def get_feature_importance(self):
        '''
        Calculates the feature importance of the model.
        '''

        # Train the XGBoost model
        model = xgb.XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.3)
        model.fit(self.X, self.y)

        # Get the feature importance from the model
        feature_importance = model.feature_importances_
        self.feature_importance = {i: feature_importance[i]*self.samples for i in range(len(feature_importance))}

    def get_binary_features(self):
        '''
        Returns a list of binary features.
        '''

        self.binary = {} # dictionary containing features with n_unique <= 2 {feature_index: n_unique}
        # get a list of binary features from self.client_X, it is a numpy array
        for i in range(self.X.shape[1]):
            unique_vals = np.unique(self.X[:, i])
            if len(unique_vals) <= 2:
                self.binary[i] = set(unique_vals)
        

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





# Move this function to the top level (outside any class or method)
def compute_histogram_for_client(client, feature_splits, compute_regions):
    return client.compute_histogram(feature_splits, compute_regions)




class FedXGBoost(XGBoostClassifier):
    
    def __init__(self):
        super().__init__()


    def fit(self, clients, avg_splits=2):
        self.clients = clients
        self.clients = clients
        self.avg_splits = avg_splits
        self.global_feature_splits = {}
        self.global_y = np.array([])

        self.initialize()


    def initialize(self):
        ''' initialize base preds and splits per feature'''
        print('Initializing Clients')

        self.feature_importance = {}
        self.quantiles = {}
        binary = {i: set() for i in range(len(self.feature_importance))}
        
        for client in self.clients:
            self.global_y = np.concatenate([self.global_y, client.y])
            if not self.feature_importance:
                self.feature_importance = client.feature_importance
                self.quantiles = client.quantiles
            else:
                for feature in client.feature_importance:
                    self.feature_importance[feature] += client.feature_importance[feature]
                    self.quantiles[feature] += client.quantiles[feature]


            binary = {i: set() for i in range(len(self.feature_importance))}
        for client in self.clients:
            for feature in client.binary:
                binary[feature] = binary[feature].union(client.binary[feature])
                
        self.binary = {}
        for feature in binary:
            if len(binary[feature]) == 2:
                self.binary[feature] = list(binary[feature])

        
        self.base_predict()

        self.splits_per_feature = self.assign_splits(self.avg_splits)

        
        for feature in self.excluded_features:
            self.feature_importance.pop(feature)

        feature_splits = {}
        for feature, splits in self.splits_per_feature.items():
            feature_values = self.quantiles[feature]
            if feature not in self.binary:
                feature_splits[feature] = np.quantile(feature_values, q=np.linspace(0, 1, splits + 2)[1:-1]).tolist()
            else:
                feature_splits[feature] = np.mean(self.binary[feature])
        
        self.feature_splits = feature_splits

        for client in self.clients:
            client.base_y = self.base_y



    # def histogram_aggregation(self, feature_splits, compute_regions=True):
    #     """
    #     Aggregate the histograms from all clients into a global histogram.
    #     """
    #     self.global_histogram = None
        
    #     # Use ProcessPoolExecutor for parallel processing
    #     with ProcessPoolExecutor() as executor:
    #         # Parallelize histogram computation across clients
    #         client_histograms = list(executor.map(
    #             compute_histogram_for_client,
    #             self.clients,  # Iterate over all clients
    #             [feature_splits] * len(self.clients),  # Same feature_splits for all clients
    #             [compute_regions] * len(self.clients)  # Same compute_regions for all clients
    #         ))

    #     # Aggregate results from all clients
    #     for client_histogram in client_histograms:
    #         if self.global_histogram is None:
    #             self.global_histogram = client_histogram
    #         else:
    #             self.global_histogram['gradients'] += client_histogram['gradients']
    #             self.global_histogram['hessians'] += client_histogram['hessians']
    #             self.global_histogram['counts'] += client_histogram['counts']

    def histogram_aggregation(self, feature_splits, compute_regions=True):
            """
            Aggregate the histograms from all clients into a global histogram.
            """

            self.global_histogram = None
            for client in self.clients:
                client_histogram = client.compute_histogram(feature_splits, compute_regions)
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


    
    def train(self, resume = False, min_child_weight = 1, depth = 5, min_leaf = 5,
            learning_rate = 0.3, boosting_rounds = 5, lambda_ = 1.5, gamma = 1,
            features_per_booster = 10):


        self.depth = depth
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma

        if not resume:
            self.estimators = []

        for client in self.clients:
            client.learning_rate = self.learning_rate
            if not resume:
                client.estimators = []

        sampling = True

        for booster in range(self.boosting_rounds):
            if features_per_booster < len(self.feature_importance):
                features_subset = self.sample_n_features(features_per_booster, self.feature_importance)
            else:
                features_subset = list(self.feature_importance.keys())
                sampling = False

            feature_splits = {feature: self.feature_splits[feature] for feature in features_subset}
            compute_regions = False
            if booster == 0 or sampling:
                compute_regions = True
            self.histogram_aggregation(feature_splits, compute_regions)
            boosting_tree = XGBoostTree().hist_fit(self.global_histogram, feature_splits=feature_splits,
                                                   min_leaf = self.min_leaf, min_child_weight = self.min_child_weight, depth = self.depth, 
                                                   lambda_ = self.lambda_, gamma = self.gamma)
            
            self.estimators.append(boosting_tree)

            for client in self.clients:
                client.estimators.append(boosting_tree)

            print(f'Boosting round {booster + 1} done.')


        print('Training Complete')


    def predict_proba(self, X):
        # pred is a 1D array of all values equal to self.base_y of shape (X.shape[0],): [y, y, y, y ...]
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        np.full((X.shape[0], 1), self.base_y).flatten().astype('float64')

        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))
    
    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        pred = np.full((X.shape[0], 1), self.base_y).flatten().astype('float64')
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
        
        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return(preds)
    