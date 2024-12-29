import numpy as np
import math
import pandas as pd
import itertools



class Histogram:

    def __init__(self, splits_per_feature={}, feature_splits={}, binary_features={}):
        
        self.splits_per_feature = splits_per_feature
        if not splits_per_feature:
            self.splits_per_feature = {feature: len(splits) for feature, splits in feature_splits.items()}
        self.binary_features = binary_features
        self.feature_splits = feature_splits



    def fit(self, X):
        if not self.feature_splits:
            self.feature_splits = self.assign_splits(X)
        self.regions = self.get_region_indices(X)


    def assign_splits(self, X):
        """
        Assign feature splits to each feature based on the number of splits per feature.
        
        Args:
            X (numpy array): Numpy ataset.
            
        Returns:
            dict: A dictionary mapping feature names to split values.
        """
        splits_per_feature = self.splits_per_feature
        feature_splits = {}
        for feature, splits in splits_per_feature.items():
            feature_values = X[:, feature]
            if feature not in self.binary_features:
                feature_splits[feature] = np.quantile(feature_values, q=np.linspace(0, 1, splits + 2)[1:-1]).tolist()
            else:
                feature_splits[feature] = np.mean(self.binary_features[feature])
        return feature_splits


    # def get_region_indices(self, df):
    #     """
    #     Identify indices of dataset points in each region defined by feature splits,
    #     optimized to avoid Cartesian products and leverage vectorized operations.
        
    #     Args:
    #         df (pd.DataFrame): Dataset with features as columns.
    #         feature_splits (dict): Dictionary where keys are feature names and values are split values.

    #     Returns:
    #         dict: A dictionary mapping region tuples to lists of indices.
    #     """
    #     print('Optimized: Getting regions')
    #     df = df.reset_index(drop=True)
        
    #     # Prepare feature splits and features
    #     feature_splits = self.feature_splits
    #     features = list(feature_splits.keys())
        
    #     # Precompute masks for each feature's split
    #     feature_masks = {}
    #     for feature in features:
    #         splits = feature_splits[feature]
    #         # Generate masks for each split point
    #         masks = []
    #         for i, split in enumerate(splits):
    #             lower_mask = (df[feature] > splits[i - 1]) if i > 0 else np.ones(len(df), dtype=bool)
    #             upper_mask = (df[feature] <= split)
    #             masks.append(lower_mask & upper_mask)
    #         # Add a final region mask for values beyond the last split
    #         masks.append(df[feature] > splits[-1])
    #         feature_masks[feature] = masks
        
    #     # Initialize region dictionary
    #     regions = {}
        
    #     # Recursive function to combine feature masks and assign indices
    #     def combine_masks(feature_idx, current_mask, region_key):
    #         if feature_idx == len(features):
    #             # Base case: assign indices for the current region
    #             regions[tuple(region_key)] = df[current_mask].index.tolist()
    #             return
            
    #         feature = features[feature_idx]
    #         for i, mask in enumerate(feature_masks[feature]):
    #             # Combine current mask with the mask for the current feature
    #             new_mask = current_mask & mask
    #             new_region_key = region_key + [(feature, i)]
    #             combine_masks(feature_idx + 1, new_mask, new_region_key)
        
    #     # Start the recursive process
    #     combine_masks(0, np.ones(len(df), dtype=bool), [])
        
    #     return regions

    def get_region_indices(self, data):
        """
        Identify indices of dataset points in each region defined by feature splits,
        optimized to avoid Cartesian products and leverage vectorized operations.

        Args:
            data (np.ndarray): Dataset as a numpy array with features as columns.
            
        Returns:
            dict: A dictionary mapping region tuples to lists of indices.
        """
        print('Getting regions')

        # Prepare feature splits and feature indices
        feature_splits = self.feature_splits
        features = list(feature_splits.keys())

        # Precompute masks for each feature's split
        feature_masks = {}
        for feature in features:
            splits = feature_splits[feature]
            # Generate masks for each split point
            masks = []
            for i, split in enumerate(splits):
                lower_mask = (data[:, feature] > splits[i - 1]) if i > 0 else np.ones(data.shape[0], dtype=bool)
                upper_mask = (data[:, feature] <= split)
                masks.append(lower_mask & upper_mask)
            # Add a final region mask for values beyond the last split
            masks.append(data[:, feature] > splits[-1])
            feature_masks[feature] = masks

        # Initialize region dictionary
        regions = {}

        # Recursive function to combine feature masks and assign indices
        def combine_masks(feature_idx, current_mask, region_key):
            if feature_idx == len(features):
                # Base case: assign indices for the current region
                regions[tuple(region_key)] = np.where(current_mask)[0].tolist()
                return
            
            feature = features[feature_idx]
            for i, mask in enumerate(feature_masks[feature]):
                # Combine current mask with the mask for the current feature
                new_mask = current_mask & mask
                new_region_key = region_key + [(feature, i)]
                combine_masks(feature_idx + 1, new_mask, new_region_key)

        # Start the recursive process
        combine_masks(0, np.ones(data.shape[0], dtype=bool), [])

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


