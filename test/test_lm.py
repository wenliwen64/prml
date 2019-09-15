from prml.lm import LinearModel as lr
from prml.lm import LogisticRegressor as lor
from prml.constants import TestCompTol
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import copy

def get_two_feature_data_set():
    from sklearn import datasets
    iris = datasets.load_iris()
    Y = iris.target[iris.target != 2]
    X = iris.data[:len(Y), :2]  # we only take the first two features.
    return X, Y


class TestLR:
    def simple(self):
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        y = y.reshape((len(y), 1))
        reg = LinearRegression().fit(X, y)
        reg_lr_weights = lr().train(X, y)[0].reshape((1, X.shape[1] + 1))

        for idx, weight in enumerate(reg_lr_weights[:-1]):
            assert abs(weight - reg.coef_[idx]) < TestCompTol

        assert abs(reg_lr_weights[0, -1] - reg.intercept_[0]) < TestCompTol


class TestLoR:
    def test_logit(self):
        assert abs(lor.logit_f(1) - 0.7310585786300049) < TestCompTol

    def logistic_f(self):
        theta = np.array([1, 1])
        A = np.array([[1, 1], [2, 1]])
        tgt = np.array([1, 0])
        #assert abs(lor.logistic_f(theta, A, tgt)) < TestCompTol

    def test_gradient_logistic_f(self):
        tgt = np.array([0, 0]).reshape((2, 1))
        A = np.array([[10., 2.], [3., 40.]])
        x0 = np.array([0., 0.]).reshape((2, 1))
        delta = 0.000002
        delta_x1 = np.array([delta, 0.0]).reshape((2, 1))
        delta_x2 = np.array([0.0, delta]).reshape((2, 1))
        grad1 = (lor.logistic_f(x0 + delta_x1, A, tgt) - lor.logistic_f(x0, A, tgt)) / delta
        grad2 = (lor.logistic_f(x0 + delta_x2, A, tgt) - lor.logistic_f(x0, A, tgt)) / delta
        print("grad1:", grad1, lor.gradient_logistic_f(x0, A, tgt)[0])
        print("grad2:", grad2, lor.gradient_logistic_f(x0, A, tgt)[1])

    def test_simple(self):
        X, y = get_two_feature_data_set()
        y = y.reshape((len(y), 1))
        y10 = copy.copy(y)
        y[y < .5] = -1
        y11 = copy.copy(y)
        X = np.array(X)
        y = np.array(y)
        #y = np.array([1, 0]).reshape((2, 1))
        #X = np.array([[10., 2.], [3., 40.]])
        #print(X, y10, y11)
        clf0 = LogisticRegression(penalty='l2', solver='liblinear', C=4.0)
        reg = clf0.fit(X, y11)
        clf = lor()
        reg_lor_weights = clf.train(X, y11, C=4.0)[0].reshape((1, X.shape[1] + 1))

        print(reg.coef_, reg.intercept_)
        print(reg_lor_weights, reg_lor_weights.shape, reg.coef_.shape, reg.intercept_.shape)
        for idx, weight in enumerate(reg_lor_weights[0][:-1]):
            print(idx, weight)
            assert abs(weight - reg.coef_[0, idx]) < 0.001

        assert abs(reg_lor_weights[0, 2] - reg.intercept_[0]) < 0.001
        #res0 = clf0.predict(X)
        #for idx, row in enumerate(X):
        #    pred = clf.predict(np.append(row, 1.0).T)
        #    print("pred:", pred, "vs.", res0[idx], "ans:", y[idx])
        #assert abs(reg_lor_weights[0, -1] - reg.intercept_[0]) < TestCompTol
