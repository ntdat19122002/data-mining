import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=0
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self.best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)
        left = self.grow_tree(X.iloc[left_idxs, :], y.iloc[left_idxs], depth+1)
        right = self.grow_tree(X.iloc[right_idxs, :], y.iloc[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            # X_column = X[:, feat_idx]
            X_column = X.iloc[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self.information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def information_gain(self, y, X_column, threshold):
        parent_entropy = self.entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y.iloc[left_idxs]), self.entropy(y.iloc[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def entropy(self, y):
        hist = np.bincount(y.iloc[:,0].values)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def most_common_label(self, y):
        counter = Counter(y.iloc[:, 0])
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X.values.tolist()])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
        


