import numpy as np
import pandas as pd
from math import e

class Node:
    
    '''
    A node object that is recursivly called within itslef to construct a regression tree. Based on Tianqi Chen's XGBoost 
    the internal gain used to find the optimal split value uses both the gradient and hessian. Also a weighted quantlie sketch 
    and optimal leaf values all follow Chen's description in "XGBoost: A Scalable Tree Boosting System" the only thing not 
    implemented in this version is sparsity aware fitting or the ability to handle NA values with a default direction.

    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins

    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''

    def __init__(self, x, gradient, hessian, idxs, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
      
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs 
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma  = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]
        
        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])
          
        self.score = float('-inf')
        self.find_varsplit()
        
        
    def compute_gamma(self, gradient, hessian):
        '''
        Calculates the optimal leaf value equation (5) in "XGBoost: A Scalable Tree Boosting System"
        '''
        return(-np.sum(gradient)/(np.sum(hessian) + self.lambda_))
        
    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initalised at the begining then no splits further splits are made
        '''
        for c in self.column_subsample: self.find_greedy_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[lhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        self.rhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[rhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        
    def find_greedy_split(self, var_idx):
        '''
         For a given feature greedily calculates the gain at each split.
         Globally updates the best score and split point if a better split point is found
        '''
        x = self.x[self.idxs, var_idx]
        
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            
            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]
                
    def weighted_qauntile_sketch(self, var_idx):
        '''
        XGBOOST Mini-Version
        Yiyang "Joe" Zeng
        Is an approximation to the eact greedy approach faster for bigger datasets wher it is not feasible
        to calculate the gain at every split point. Uses equation (8) and (9) from "XGBoost: A Scalable Tree Boosting System"
        '''
        x = self.x[self.idxs, var_idx]
        hessian_ = self.hessian[self.idxs]
        df = pd.DataFrame({'feature':x,'hess':hessian_})
        
        df.sort_values(by=['feature'], ascending = True, inplace = True)
        hess_sum = df['hess'].sum() 
        df['rank'] = df.apply(lambda x : (1/hess_sum)*sum(df[df['feature'] < x['feature']]['hess']), axis=1)
        
        for row in range(df.shape[0]-1):
            # look at the current rank and the next ran
            rk_sk_j, rk_sk_j_1 = df['rank'].iloc[row:row+2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if(diff >= self.eps):
                continue
                
            split_value = (df['rank'].iloc[row+1] + df['rank'].iloc[row])/2
            lhs = x <= split_value
            rhs = x > split_value
            
            lhs_indices = np.nonzero(x <= split_value)[0]
            rhs_indices = np.nonzero(x > split_value)[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight): continue
                
            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value
                
    def gain(self, lhs, rhs):
        '''
        Calculates the gain at a particular split point bases on equation (7) from
        "XGBoost: A Scalable Tree Boosting System"
        '''
        gradient = self.gradient[self.idxs]
        hessian  = self.hessian[self.idxs]
        
        lhs_gradient = gradient[lhs].sum()
        lhs_hessian  = hessian[lhs].sum()
        
        rhs_gradient = gradient[rhs].sum()
        rhs_hessian  = hessian[rhs].sum()
        
        gain = 0.5 *( (lhs_gradient**2/(lhs_hessian + self.lambda_)) + (rhs_gradient**2/(rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient)**2/(lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return(gain)
                
    @property
    def split_col(self):
        '''
        splits a column 
        '''
        return self.x[self.idxs , self.var_idx]
                
    @property
    def is_leaf(self):
        '''
        checks if node is a leaf
        '''
        return self.score == float('-inf') or self.depth <= 0                 

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return(self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)
    

class HistogramNode:
    def __init__(self, hist_data, feature_names, depth=10, min_leaf=5, min_child_weight=1, lambda_=1, gamma=1):
        """
        Initializes a tree node using histogram data for XGBoost-style training.
        
        Parameters:
        - hist_data: Precomputed histogram data containing 'values', 'gradient', 'hessian', and 'bin_edges' for each feature.
        - feature_names: List of feature names corresponding to the hist_data.
        - depth: Maximum depth of the tree.
        - min_leaf: Minimum number of samples required in a leaf node.
        - min_child_weight: Minimum sum of Hessians required for a split.
        - lambda_: Regularization parameter for leaf weights.
        - gamma: Minimum gain required to make a split.
        """
        self.hist_data = hist_data
        self.feature_names = feature_names
        self.depth = depth
        self.min_leaf = min_leaf
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma

        # Node-specific properties
        self.score = float('-inf')
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.val = self.compute_leaf_value()

        # Build the tree
        self.find_best_split()

    def compute_leaf_value(self):
        """
        Computes the optimal value for the current node's leaf.
        """
        total_grad = sum(hist['gradient'].sum() for hist in self.hist_data.values())
        total_hess = sum(hist['hessian'].sum() for hist in self.hist_data.values())
        return -total_grad / (total_hess + self.lambda_)

    def find_best_split(self):
        """
        Finds the best split for the current node using the histogram data.
        """
        if self.depth <= 0:
            return

        for feature in self.feature_names:
            hist = self.hist_data[feature]
            gradients = hist['gradient']
            hessians = hist['hessian']
            bin_edges = hist['bin_edges']

            for bin_idx in range(len(bin_edges) - 1):
                lhs_hess = hessians[:bin_idx + 1].sum()
                rhs_hess = hessians[bin_idx + 1:].sum()

                if lhs_hess < self.min_child_weight or rhs_hess < self.min_child_weight:
                    continue

                lhs_grad = gradients[:bin_idx + 1].sum()
                rhs_grad = gradients[bin_idx + 1:].sum()

                gain = 0.5 * (
                    (lhs_grad ** 2 / (lhs_hess + self.lambda_))
                    + (rhs_grad ** 2 / (rhs_hess + self.lambda_))
                    - ((lhs_grad + rhs_grad) ** 2 / (lhs_hess + rhs_hess + self.lambda_))
                ) - self.gamma

                if gain > self.score:
                    self.score = gain
                    self.split_feature = feature
                    self.split_value = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2

        # Stop splitting if no valid split is found
        if self.score == float('-inf'):
            return

        self.split_node()

    def split_node(self):
        """
        Splits the current node into left and right children.
        """
        if self.depth <= 0 or self.score == float('-inf'):
            return

        left_hist_data = {}
        right_hist_data = {}

        for feature in self.feature_names:
            hist = self.hist_data[feature]
            bin_edges = hist['bin_edges']
            split_idx = np.searchsorted(bin_edges, self.split_value, side="right") - 1

            left_hist_data[feature] = {
                'gradient': hist['gradient'][:split_idx + 1],
                'hessian': hist['hessian'][:split_idx + 1],
                'bin_edges': bin_edges[:split_idx + 2],
            }

            right_hist_data[feature] = {
                'gradient': hist['gradient'][split_idx + 1:],
                'hessian': hist['hessian'][split_idx + 1:],
                'bin_edges': bin_edges[split_idx + 1:],
            }

        self.left = HistogramNode(
            left_hist_data, self.feature_names, self.depth - 1, self.min_leaf, self.min_child_weight, self.lambda_, self.gamma
        )
        self.right = HistogramNode(
            right_hist_data, self.feature_names, self.depth - 1, self.min_leaf, self.min_child_weight, self.lambda_, self.gamma
        )

    def predict(self, X):
        """
        Predicts the output for a DataFrame of input instances.
        """
        return np.array([self.predict_row(xi) for _, xi in X.iterrows()])  # Iterating through DataFrame rows

    def predict_row(self, xi):
        """
        Predicts a single instance based on the tree structure.
        """
        if self.left is None or self.right is None:
            return self.val

        # Ensure xi is a pandas Series (it should be when iterating with DataFrame.iterrows())
        if xi[self.split_feature] <= self.split_value:
            return self.left.predict_row(xi)
        else:
            return self.right.predict_row(xi)



    
class XGBoostTree:
    '''
    Wrapper class that provides a scikit learn interface to the recursive regression tree above
    
    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins
    
    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    
    '''
    def fit(self, x, gradient, hessian, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self
    
    def hist_fit(self, histograms, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1, gamma=1, eps=0.1):
        """
        Fits a regression tree using precomputed histograms for XGBoost-style training.

        Parameters:
        - histograms: A dictionary where each key is a feature name, and the value is a dictionary containing
                    'values', 'gradient', 'hessian', and 'bin_edges' for that feature.
        - subsample_cols: Fraction of features to randomly sample for splits at each level.
        - min_leaf: Minimum number of samples required in a leaf node.
        - min_child_weight: Minimum sum of Hessians required for a split.
        - depth: Maximum depth of the tree.
        - lambda_: Regularization parameter for leaf weights.
        - gamma: Minimum gain required to make a split.
        - eps: Used for quantile sketching (not needed for histograms but kept for interface consistency).

        Returns:
        - self: The fitted tree.
        """
        # Subsample columns for each level of the tree
        feature_names = list(histograms.keys())
        num_features = len(feature_names)
        num_selected_features = max(1, int(subsample_cols * num_features))
        selected_features = np.random.choice(feature_names, num_selected_features, replace=False)

        # Filter histograms to only include selected features
        selected_histograms = {feature: histograms[feature] for feature in selected_features}

        # Build the tree using HistogramNode
        self.dtree = HistogramNode(
            hist_data=selected_histograms,
            feature_names=selected_features,
            depth=depth,
            min_leaf=min_leaf,
            min_child_weight=min_child_weight,
            lambda_=lambda_,
            gamma=gamma
        )
        return self

    
    def predict(self, X):
        return self.dtree.predict(X)
   
   
class XGBoostClassifier:
    '''
    Full application of the XGBoost algorithm as described in "XGBoost: A Scalable Tree Boosting System" for 
    Binary Classification.

    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins

    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''
    def __init__(self, method = 'regular'):
        self.estimators = []
        self.method = method
    
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
    
    @staticmethod
    def log_odds(column):
        binary_yes = np.count_nonzero(column == 1)
        binary_no  = np.count_nonzero(column == 0)
        return(np.log(binary_yes/binary_no))
    
    
    def fit(self, X, y, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):

        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma
    
        self.base_pred = np.full((X.shape[0], 1), 0).flatten().astype('float64')
        
        if self.method == 'regular':
            X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
            y = y.to_numpy() if isinstance(y, pd.Series) else y
            self.X, self.y = X, y
            for booster in range(self.boosting_rounds):
                Grad = self.grad(self.base_pred, self.y)
                Hess = self.hess(self.base_pred, self.y)
                boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
                self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
                self.estimators.append(boosting_tree)

        elif self.method == 'hist':
            self.max_bin = 256
            self.hist_data = {}
            for booster in range(self.boosting_rounds):
                self.compute_histograms(X, y.to_numpy(), self.base_pred, max_bins=self.max_bin)
                boosting_tree = XGBoostTree().hist_fit(self.hist_data, subsample_cols = self.subsample_cols, min_leaf = self.min_leaf, min_child_weight = self.min_child_weight, depth = self.depth, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps)
                self.base_pred += self.learning_rate * boosting_tree.predict(X)
                self.estimators.append(boosting_tree)

        print('Training Complete')




    def compute_histograms(self, X, y, y_preds, max_bins=256):



        for column in X.columns:
            # Compute quantile-based bins
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
                'bin_edges': bin_edges
            }

          
    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))
    
    def predict(self, X):
        if self.method == 'regular':
            X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
        
        predicted_probas = self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return(preds)
       
       
class XGBoostRegressor:
    '''
    Full application of the XGBoost algorithm as described in "XGBoost: A Scalable Tree Boosting System" for 
    regression.

    Inputs
    ------------------------------------------------------------------------------------------------------------------
    x: pandas datframe of the training data
    gradient: negative gradient of the loss function
    hessian: second order derivative of the loss function
    idxs: used to keep track of samples within the tree structure
    subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
    (complexity parameter)
    min_leaf: minimum number of samples for a node to be considered a node (complexity parameter)
    min_child_weight: sum of the heassian inside a node is a meaure of purity (complexity parameter)
    depth: limits the number of layers in the tree
    lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
    gamma: This parameter also prevents over fitting and is present in the the calculation of the gain (structure score). 
    As this is subtracted from the gain it essentially sets a minimum gain amount to make a split in a node.
    eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
    (1 / sketch_eps) number of bins

    Outputs
    --------------------------------------------------------------------------------------------------------------------
    A single tree object that will be used for gradient boosintg.
    '''
    def __init__(self):
        self.estimators = []
    
    # first order gradient mean squared error
    @staticmethod
    def grad(preds, labels):
        return(2*(preds-labels))
    
    # second order gradient logLoss
    @staticmethod
    def hess(preds, labels):
        '''
        hessian of mean squared error is a constant value of two 
        returns an array of twos
        '''
        return(np.full((preds.shape[0], 1), 2).flatten().astype('float64'))
    
    
    def fit(self, X, y, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma
    
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten().astype('float64')
    
        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)
          
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return np.full((X.shape[0], 1), np.mean(self.y)).flatten().astype('float64') + pred