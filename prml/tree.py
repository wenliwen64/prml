import numpy as np
from prml.utils import cross_entropy, gini
from collections import Counter

def _single_class_stop_f(node):
    return len(Counter(node.tgts)) == 1


def _find_optimal_cut_point(X_y, score_f):
    feature_and_tgt = X_y[np.lexsort((X_y[:, 0],))]

    total_samples = len(feature_and_tgt)
    optimal_impurity_reduction, optimal_i_cutpoint = -1.0, -1
    for i_cutpoint, (feature, tgt) in enumerate(feature_and_tgt[1:], start=1):
        if tgt != feature_and_tgt[i_cutpoint - 1, -1]:
            impurity_reduction_tmp = score_f(feature_and_tgt[:, -1]) \
              - score_f(feature_and_tgt[:i_cutpoint, -1]) * i_cutpoint / total_samples\
              - score_f(feature_and_tgt[i_cutpoint:, -1]) * (total_samples - i_cutpoint) / total_samples
            if impurity_reduction_tmp > optimal_impurity_reduction:
                optimal_impurity_reduction = impurity_reduction_tmp
                optimal_i_cutpoint = i_cutpoint
    return (optimal_impurity_reduction, .5 * (feature_and_tgt[optimal_i_cutpoint, 0] +
                                              feature_and_tgt[optimal_i_cutpoint - 1, 0]))

class DecisionTreeNode:
    def __init__(self, observations, tgts):
        self.samples = observations
        self.tgts = tgts
        self.left, self.right = None, None
        (self.optimal_impurity_reduction, self.optimal_feature_i, self.optimal_cutpoint) = \
        (None,                            None,                     None)

    @property
    def majority(self):
        return Counter(self.tgts).most_common()[0][0]

    @property
    def total_samples_num(self):
        return len(self.samples)

    @property
    def num_classes(self):
        return len(Counter(self.tgts))

    def grow_a_tree(self, stop_f, score_f):
        if (stop_f(self)):
            return self

        (self.optimal_impurity_reduction, self.optimal_feature_i, self.optimal_cutpoint) = \
        (-1.0,                            -1,                     None)
        obs_and_tgt = np.append(self.samples, self.tgts.reshape(len(self.samples), 1), axis=1)
        print(obs_and_tgt.shape)

        for i_feature in range(obs_and_tgt.shape[1] - 1):
            optimal_impurity_reduction_tmp, optimal_cutpoint_tmp =\
                _find_optimal_cut_point(obs_and_tgt[:, [i_feature, -1]], score_f)

            if optimal_impurity_reduction_tmp > self.optimal_impurity_reduction:
                (self.optimal_impurity_reduction, self.optimal_feature_i, self.optimal_cutpoint) = \
                (optimal_impurity_reduction_tmp,  i_feature,              optimal_cutpoint_tmp)

        obs_and_tgt = obs_and_tgt[np.lexsort((obs_and_tgt[:, self.optimal_feature_i],))]

        left_obs_and_tgt = obs_and_tgt[obs_and_tgt[:, self.optimal_feature_i] < self.optimal_cutpoint, :]
        right_obs_and_tgt = obs_and_tgt[obs_and_tgt[:, self.optimal_feature_i] >= self.optimal_cutpoint, :]

        self.left = DecisionTreeNode(observations=left_obs_and_tgt[:, :-1],
                                     tgts=left_obs_and_tgt[:, -1]).grow_a_tree(stop_f, score_f)
        self.right = DecisionTreeNode(observations=right_obs_and_tgt[:, :-1],
                                      tgts=right_obs_and_tgt[:, -1]).grow_a_tree(stop_f, score_f)
        return self

    def _predict_single(self, x):
        # terminal node?
        if self.left is None:
            return self.majority

        if x[self.optimal_feature_i] < self.optimal_cutpoint :
            return self.left._predict_single(x)
        else:
            return self.right._predict_single(x)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _print(self):
        #print(self.optimal_feature_i, self.optimal_cutpoint, self.majority)
        if self.left is not None:
            self.left._print()
            self.right._print()


class DecisionTree:
    def __init__(self, stop_f=_single_class_stop_f, score_f=cross_entropy):
        self.stop_f = stop_f
        self.score_f = score_f

    def train(self, X, tgts):
        self.root = DecisionTreeNode(X, tgts)
        self.root.grow_a_tree(self.stop_f, self.score_f)

    def predict(self, X):
        return self.root.predict(X)

    def _debug(self):
        self.root._print()
