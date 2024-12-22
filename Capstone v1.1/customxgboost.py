import numpy as np
import pandas as pd
from math import e
from histogram import Histogram

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

    def __init__(self, x, gradient, hessian, idxs, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1, 
                 depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
      
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
    """
    A node object that constructs a regression tree using histograms of gradients and hessians.
    This class is designed for use with a histogram-based implementation of XGBoost.

    Inputs
    ------------------------------------------------------------------------------------------------------------------
    histogram: A dictionary with the structure:
               histogram[feature] = {
                   'gradient': gradients (numpy array),
                   'hessian': hessians (numpy array),
                   'bin_edges': bin_edges (numpy array)
               }
    idxs: Array of sample indices associated with the node.
    subsample_cols: Proportion of columns to subsample at each layer.
    min_leaf: Minimum number of samples for a node to be considered a valid node.
    min_child_weight: Minimum sum of hessian values required in a node to allow splitting.
    depth: Maximum depth of the tree.
    lambda_: L2 regularization term on weights.
    gamma: Minimum gain required to make a split.
    """

    def __init__(self, histogram, feature_splits, subsample_cols=0.8, min_leaf=5, min_child_weight=1,
                 depth=10, lambda_=1, gamma=1):
        self.histogram = histogram
        self.feature_splits = feature_splits
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight

        self.col_count = len(feature_splits)
        self.subsample_cols = subsample_cols

        self.column_subsample = np.random.permutation(self.col_count)[:
            round(self.subsample_cols * self.col_count)
        ]

        self.val = self.compute_gamma()
        self.score = float('-inf')
        self.find_varsplit()

    def compute_gamma(self):
        """
        Calculates the optimal leaf value using the sum of gradients and hessians.
        """
        total_gradient = np.sum(self.histogram['gradients'])
        total_hessian = np.sum(self.histogram['hessians'])
        return -total_gradient / (total_hessian + self.lambda_)

    def find_varsplit(self):
        """
        Identifies the best feature and split point for splitting the node.
        """
        for feature_idx in self.column_subsample:
            self.find_greedy_split(feature_idx)

        if self.is_leaf:
            return


        features = list(self.feature_splits.keys())
        feature = features[self.var_idx]

        # Get the left and right idxs based on the split
        lhs_feature_splits = self.feature_splits.copy()
        lhs_feature_splits[feature] = lhs_feature_splits[feature][:self.split_idx]
        rhs_feature_splits = self.feature_splits.copy()
        rhs_feature_splits[feature] = rhs_feature_splits[feature][self.split_idx + 1:]

        self.lhs = HistogramNode(
            histogram=self.left_child_hist,
            feature_splits=lhs_feature_splits,
            subsample_cols=self.subsample_cols,
            min_leaf=self.min_leaf,
            min_child_weight=self.min_child_weight,
            depth=self.depth - 1,
            lambda_=self.lambda_,
            gamma=self.gamma
        )
        self.rhs = HistogramNode(
            histogram=self.right_child_hist,
            feature_splits=rhs_feature_splits,
            subsample_cols=self.subsample_cols,
            min_leaf=self.min_leaf,
            min_child_weight=self.min_child_weight,
            depth=self.depth - 1,
            lambda_=self.lambda_,
            gamma=self.gamma
        )

    def find_greedy_split(self, feature_idx):
        """
        Evaluates possible split points for a given feature using the histogram.
        """
        gradients = self.histogram['gradients']
        hessians = self.histogram['hessians']
        counts = self.histogram['counts']

        no_of_splits = len(list(self.feature_splits.values())[feature_idx])
        for split_idx in range(no_of_splits):
            lhs_gradients, rhs_gradients = self.split_feature(gradients, feature_idx, split_idx+1)
            lhs_hessians, rhs_hessians = self.split_feature(hessians, feature_idx, split_idx+1)
            lhs_counts, rhs_counts = self.split_feature(counts, feature_idx, split_idx+1)

            lhs_gradient, rhs_gradient = np.sum(lhs_gradients), np.sum(rhs_gradients)
            lhs_hessian, rhs_hessian = np.sum(lhs_hessians), np.sum(rhs_hessians)
            lhs_count, rhs_count = np.sum(lhs_counts), np.sum(rhs_counts)

            if (lhs_hessian < self.min_child_weight or
                rhs_hessian < self.min_child_weight or
                lhs_count < self.min_leaf or rhs_count < self.min_leaf):
                continue

            gain = self.compute_gain(lhs_gradient, lhs_hessian, rhs_gradient, rhs_hessian)

            if gain > self.score:
                self.var_idx = feature_idx
                self.score = gain
                self.split_idx = split_idx

                self.left_child_hist = {'gradients': lhs_gradients,
                                        'hessians': lhs_hessians,
                                        'counts': lhs_counts}
                
                self.right_child_hist = {'gradients': rhs_gradients,
                                        'hessians': rhs_hessians,
                                        'counts': rhs_counts}                

    def compute_gain(self, lhs_gradient, lhs_hessian, rhs_gradient, rhs_hessian):
        """
        Computes the gain for a given split based on equation (7) from the XGBoost paper.
        """
        gain = 0.5 * (
            (lhs_gradient ** 2 / (lhs_hessian + self.lambda_)) +
            (rhs_gradient ** 2 / (rhs_hessian + self.lambda_)) -
            ((lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + self.lambda_))
        ) - self.gamma
        return gain
    
    @staticmethod
    def split_feature(array, axis, index):
        '''
        splits a multidensional array at index 'index' of the 'axis' dimension
        '''

        split1 = array.take(indices=range(0, index), axis=axis)  # First part
        split2 = array.take(indices=range(index, array.shape[axis]), axis=axis)

        return split1, split2

    @property
    def is_leaf(self):
        """
        Determines if the current node is a leaf.
        """
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        
        split_value = list(self.feature_splits.values())[self.var_idx][self.split_idx]

        node = self.lhs if xi[self.var_idx] <= split_value else self.rhs
        return node.predict_row(xi)




    
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
    
    def hist_fit(self, histogram, max_bins, feature_splits = None, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1, gamma=1):
        if feature_splits is None:
            feature_splits = histogram.feature_splits
        if isinstance(histogram, Histogram):
            histogram = histogram.histogram
        self.dtree = HistogramNode(histogram, feature_splits, subsample_cols=0.8, min_leaf=5, min_child_weight=1,
                 depth=10, lambda_=1, gamma=1)
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
        self.max_bins = None
    
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

        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.Series) else y       

        Grad = self.grad(self.base_pred, self.y)
        Hess = self.hess(self.base_pred, self.y)
        
        if self.method == 'regular':
            for booster in range(self.boosting_rounds):
                boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
                self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
                self.estimators.append(boosting_tree)

        elif self.method == 'hist':
            max_bins = 256 if self.max_bins is None else self.max_bins
            histogram = Histogram(max_bins=max_bins)
            histogram.fit(X)
            for booster in range(self.boosting_rounds):
                histogram.compute_histogram(Grad, Hess)
                boosting_tree = XGBoostTree().hist_fit(histogram, max_bins, subsample_cols = self.subsample_cols, min_leaf = self.min_leaf, min_child_weight = self.min_child_weight, depth = self.depth, lambda_ = self.lambda_, gamma = self.gamma)
                self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
                self.estimators.append(boosting_tree)

        print('Training Complete')


          
    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(np.full((X.shape[0], 1), 1).flatten().astype('float64') + pred))
    
    def predict(self, X):
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