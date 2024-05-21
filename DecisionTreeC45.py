import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeC45:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y)

        # Create child nodes
        left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)
        left = self._grow_tree(X.iloc[left_idxs, :], y.iloc[left_idxs], depth + 1)
        right = self._grow_tree(X.iloc[right_idxs, :], y.iloc[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y):
        best_gain_ratio = -1
        split_idx, split_value = None, None

        for feat_idx in range(X.shape[1]):
            unique_values = np.unique(X.iloc[:, feat_idx])
            if len(unique_values) == 1:  # Skip if only one unique value
                continue

            for value in unique_values:
                gain_ratio = self._gain_ratio(X.iloc[:, feat_idx], y, value)

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    split_idx = feat_idx
                    split_value = value

        return split_idx, split_value

    def _gain_ratio(self, feature, y, value):
        n = len(y)
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(feature, value)
        n_l, n_r = len(left_idxs), len(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0

        # Information gain
        e_l, e_r = self._entropy(y.iloc[left_idxs]), self._entropy(y.iloc[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Split entropy
        split_entropy = -((n_l / n) * np.log2(n_l / n) + (n_r / n) * np.log2(n_r / n))

        # Gain Ratio
        gain = parent_entropy - child_entropy
        gain_ratio = gain / split_entropy
        return gain_ratio

    def _split(self, feature, value):
        left_idxs = np.where(feature <= value)[0]
        right_idxs = np.where(feature > value)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y.iloc[:,0].values)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y.iloc[:,0].values)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X.values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
