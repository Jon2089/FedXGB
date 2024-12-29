import numpy as np
import math
import pandas as pd
import itertools



class Histogram:

    def __init__(self, max_bins=256, client_initiate=False, feature_splits=None):
        self.max_bins = max_bins
        self.client_initiate = client_initiate
        self.feature_splits = feature_splits
        self.regions = None



    def fit(self, X):
        self.features = [feature for feature in X.columns]
        self.dim = len(self.features)
        self.max_bins = self.max_bins

        if self.feature_splits is None:
            self.feature_splits = {feature: [] for feature in self.features} # {feature: bin_edges}
            self.assign_bins(X)
        
        self.splits_per_feature = {feature: len(x) for feature, x in self.feature_splits.items()} #{feature: total splits in that feature}

        if not self.client_initiate and self.regions is None:
            self.regions = self.get_region_indices(X)


        self.fitted = True # so that histogram indices (regions) do not need to be found again

    def assign_bins(self, df):

        '''Assigns bins to column, binary and ordinal get bins = their values'''

        continuous = []
        non_continous = {}
        for column in df.columns:
            unique_values = df[column].dropna().unique()  # Exclude NaN values for classification
            n_unique = len(unique_values)
            
            if pd.api.types.is_numeric_dtype(df[column]):
                if n_unique < max(5, math.log(self.max_bins) // math.log(self.dim)):  # Arbitrary threshold for binary and ordinal
                    self.feature_splits[column] = unique_values
                    non_continous[column] = n_unique
                else:
                    continuous.append(column)

        bins_left = self.max_bins
        for column in non_continous:
            bins_left = bins_left // non_continous[column]

        bins = self.find_closest_factors(bins_left, len(continuous))

        for i, column in enumerate(continuous):
            total_bins = bins[i]
            bin_edges = np.quantile(df[column], q=np.linspace(0, 1, max(total_bins + 1, 3))) # if less than 3 then it becomes 0 below
            self.feature_splits[column] = bin_edges.tolist()[1:-1] # do not want min and max values as splits

    
    @staticmethod
    def find_closest_factors(x, n):
        '''To find numer of bins for each feature'''
        
        # Take the n-th root of x
        base = x ** (1 / n)
        
        # Split into integers close to the n-th root
        lower = math.floor(base)
        upper = math.ceil(base)
        
        # Start with all lower values
        factors = [lower] * n
        
        # Adjust the factors to make the product closer to x
        product = lower ** n
        i = 0
        while product < x and i < n:
            factors[i] = upper
            product = math.prod(factors)
            i += 1
        
        return factors


    # def get_region_indices(self, df):
    #     """
    #     Identify indices of dataset points in each region defined by feature splits.

    #     Args:
    #         df (pd.DataFrame): Dataset with features as columns.
    #         feature_splits (dict): Dictionary where keys are feature names and values are split values.

    #     Returns:
    #         dict: A dictionary mapping region tuples to lists of indices.
    #     """
    #     print('getting regions')
    #     df = df.reset_index(drop=True)
        
    #     # Generate region bounds using combinations of split values
    #     feature_splits = self.feature_splits
    #     regions = {}
    #     features = list(feature_splits.keys())
    #     splits = feature_splits.values()

    #     # Generate all possible regions by taking product of split ranges
    #     all_combinations = list(itertools.product(*[
    #         zip([None] + values, values + [None]) for values in splits
    #     ]))
        
    #     # Iterate through each region definition
    #     for region in all_combinations:
    #         # Create boolean masks for the region
    #         region_conditions = []
    #         for i, (lower, upper) in enumerate(region):
    #             feature = features[i]
    #             if lower is not None:
    #                 region_conditions.append(df[feature] > lower)
    #             if upper is not None:
    #                 region_conditions.append(df[feature] <= upper)
            
    #         # Combine all conditions using logical AND
    #         if region_conditions:
    #             mask = region_conditions[0]
    #             for condition in region_conditions[1:]:
    #                 mask = mask & condition
    #         else:
    #             mask = pd.Series(True, index=df.index)
            
    #         # Store indices of points in this region
    #         region_key = tuple(region)
    #         regions[region_key] = df[mask].index.tolist()
        
    #     return regions



    def get_region_indices(self, df):
        """
        Identify indices of dataset points in each region defined by feature splits,
        optimized to avoid Cartesian products and leverage vectorized operations.
        
        Args:
            df (pd.DataFrame): Dataset with features as columns.
            feature_splits (dict): Dictionary where keys are feature names and values are split values.

        Returns:
            dict: A dictionary mapping region tuples to lists of indices.
        """
        print('Optimized: Getting regions')
        df = df.reset_index(drop=True)
        
        # Prepare feature splits and features
        feature_splits = self.feature_splits
        features = list(feature_splits.keys())
        
        # Precompute masks for each feature's split
        feature_masks = {}
        for feature in features:
            splits = feature_splits[feature]
            # Generate masks for each split point
            masks = []
            for i, split in enumerate(splits):
                lower_mask = (df[feature] > splits[i - 1]) if i > 0 else np.ones(len(df), dtype=bool)
                upper_mask = (df[feature] <= split)
                masks.append(lower_mask & upper_mask)
            # Add a final region mask for values beyond the last split
            masks.append(df[feature] > splits[-1])
            feature_masks[feature] = masks
        
        # Initialize region dictionary
        regions = {}
        
        # Recursive function to combine feature masks and assign indices
        def combine_masks(feature_idx, current_mask, region_key):
            if feature_idx == len(features):
                # Base case: assign indices for the current region
                regions[tuple(region_key)] = df[current_mask].index.tolist()
                return
            
            feature = features[feature_idx]
            for i, mask in enumerate(feature_masks[feature]):
                # Combine current mask with the mask for the current feature
                new_mask = current_mask & mask
                new_region_key = region_key + [(feature, i)]
                combine_masks(feature_idx + 1, new_mask, new_region_key)
        
        # Start the recursive process
        combine_masks(0, np.ones(len(df), dtype=bool), [])
        
        return regions


    def compute_histogram(self, Grad, Hess):

        total_regions = len(self.regions)

        gradients = np.zeros(total_regions)
        hessians = np.zeros(total_regions)
        counts = np.zeros(total_regions)

        for i, (region, indices) in enumerate(self.regions.items()):
            gradients[i] = np.sum(Grad[indices])
            hessians[i] = np.sum(Hess[indices])
            counts[i] = len(indices)

        shape = np.array(list(self.splits_per_feature.values())) + 1 # regions = splits + 1
        gradients = gradients.reshape(shape)
        hessians = hessians.reshape(shape)
        counts = counts.reshape(shape)

        self.histogram = {'gradients': gradients, 'hessians': hessians, 'counts': counts}
        
        return self.histogram


