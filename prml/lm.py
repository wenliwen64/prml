import numpy as np
from .optimizer import gradient_descent, newton_method
from .constants import TOL
from functools import partial


class LinearModel:
    def __init__(self):
        self.weights = None

    @staticmethod
    def linear_f(weight, A, tgt):
        return np.linalg.norm(np.dot(A, weight) - tgt, ord=2) ** 2

    @staticmethod
    def gradient_linear_f(weight, A, tgt):
        # w0 + x1w1 + ... + xNwN
        return 2 * np.dot(A.T, np.dot(A, weight) - tgt)

    @staticmethod
    def hessian_linear_f(weight, A):
        return 2 * np.dot(A.T, A)

    def train(self, A_o, tgt):
        assert A_o.shape[0] == tgt.shape[0]
        # number of samples
        n = A_o.shape[0]
        # number of features
        m = A_o.shape[1] + 1

        A = np.ones([n, m])
        A[:, :-1] = A_o
        tgt = tgt.reshape((n, 1))

        self.weights = newton_method(f=partial(self.linear_f, A=A, tgt=tgt),
                                     gradient_f=partial(self.gradient_linear_f, A=A, tgt=tgt),
                                     hessian_f=partial(self.hessian_linear_f, A=A),
                                     initial_p=np.zeros((m, 1)), epsilon=TOL, dom=lambda x: True)
        return self.weights


class LogisticRegressor:
    """
    """
    def __init__(self):
        pass

    @staticmethod
    def logit_f(x):
        return np.exp(x) / (1. + np.exp(x))

    @staticmethod
    def logit_gradient_f(x):
        return np.exp(x) / ((1. + np.exp(x)) ** 2)

    @staticmethod
    def sklearn_obj_f(theta, A, tgt):
        res = 0.0
        for idx, data_entry in enumerate(A):
            res = res + np.log(1 + np.exp(-tgt[idx] * (np.dot(data_entry, theta))))
        return res

    @staticmethod
    def sklearn_obj_F(theta, A, tgt, C):
        return C * LogisticRegressor.sklearn_obj_f(theta, A, tgt) + .5 * np.linalg.norm(theta, ord=2) ** 2

    @staticmethod
    def sklearn_obj_gradient_f(theta, A, tgt):
        res = np.zeros([A.shape[1], 1])
        for idx, data_entry in enumerate(A):
            data_entry = data_entry.reshape((1, len(theta)))
            po = -tgt[idx] * np.dot(data_entry, theta)
            res = res - tgt[idx] * data_entry.T * np.exp(po) / (1 + np.exp(po))
        #print(res.shape, theta.shape)
        assert res.shape == theta.shape
        return res

    @staticmethod
    def sklearn_obj_gradient_F(theta, A, tgt, C):
        return C * LogisticRegressor.sklearn_obj_gradient_f(theta, A, tgt) + theta

    @staticmethod
    def sklearn_obj_hessian_f(theta, A, tgt):
        res = np.zeros([A.shape[1], A.shape[1]])
        for idx, data_entry in enumerate(A):
            data_entry = data_entry.reshape([1, A.shape[1]])
            z = -tgt[idx] * np.dot(data_entry, theta)
            g_second = np.exp(z) / ((1 + np.exp(z)) ** 2)
            #grad = LogisticRegressor.sklearn_obj_gradient_f(theta, A, tgt)
            res = res + g_second * np.dot(data_entry.T, data_entry) * (tgt[idx] ** 2)
        return res

    @staticmethod
    def sklearn_obj_hessian_F(theta, A, tgt, C):
        return C * LogisticRegressor.sklearn_obj_hessian_f(theta, A, tgt) + np.eye(len(theta))

    @staticmethod
    def logistic_f(theta, A, tgt):
        res = 0.0
        for idx, data_entry in enumerate(A):
            res = res + tgt[idx] * np.log(1. + np.exp(-np.dot(data_entry, theta)))\
                  + (1. - tgt[idx]) * np.log(1. + np.exp(np.dot(data_entry, theta)))
        return res

    @staticmethod
    def logistic_F(theta, A, tgt, C):
        return C * LogisticRegressor.logistic_f(theta, A, tgt) + .5 * np.linalg.norm(theta, ord=2) ** 2

    @staticmethod
    def gradient_logistic_f(theta, A, tgt):
        res = np.zeros([A.shape[1], 1])
        for idx, data_entry in enumerate(A):
            z = np.dot(data_entry, theta)
            res = res + data_entry.reshape((len(theta), 1)) * (LogisticRegressor.logit_f(z) - tgt[idx])
            #print("compute gradient:\n", data_entry.reshape((len(theta), 1)) * (LogisticRegressor.logit_f(z) - tgt[idx]), z)
        #print("gradient:", res)
        return res

    @staticmethod
    def gradient_logistic_F(theta, A, tgt, C):
        return C * LogisticRegressor.gradient_logistic_f(theta, A, tgt) + theta

    @staticmethod
    def hessian_logistic_f(theta, A, tgt):
        res = np.zeros((A.shape[1], A.shape[1]))
        #print("result.shape", res.shape)
        for idx, data_entry in enumerate(A):
            data_entry = data_entry.reshape([1, A.shape[1]])
            z = np.dot(data_entry, theta)
            g_second = np.exp(z) / ((1 + np.exp(z)) ** 2)
            #grad = LogisticRegressor.gradient_logistic_f(theta, data_entry, tgt)
            #print(data_entry.shape, z.shape, g_second.shape, grad.shape, grad.T.shape)
            res = res + g_second * np.dot(data_entry.T, data_entry)
            #print("ele: ", g_second * np.dot(grad, grad.T), "g_second", g_second, "dot:", np.dot(grad, grad.T))
        #print("hessian matrix:", res)
        #print("eigs:", np.linalg.eigvals(res))
        return res

    @staticmethod
    def hessian_logistic_F(theta, A, tgt, C):
        #print(theta.shape, A.shape, tgt.shape)
        return C * LogisticRegressor.hessian_logistic_f(theta, A, tgt) + np.eye(len(theta))

    def train(self, A_o, tgt, C):
        assert A_o.shape[0] == tgt.shape[0]
        # number of samples
        n = A_o.shape[0]
        # number of features
        m = A_o.shape[1] + 1

        A = np.ones([n, m])
        A[:, :-1] = A_o
        #print("A.shape", A.shape)
        tgt = tgt.reshape((n, 1))

        #self.weights = newton_method(f=partial(self.logistic_F, A=A, tgt=tgt, C=C),
        #                             gradient_f=partial(self.gradient_logistic_F, A=A, tgt=tgt, C=C),
        #                             hessian_f=partial(self.hessian_logistic_F, A=A, tgt=tgt, C=C),
        #                             initial_p=np.ones((m, 1)), epsilon=TOL, dom=lambda x: True)
        self.weights = newton_method(f=partial(self.sklearn_obj_F, A=A, tgt=tgt, C=C),
                                     gradient_f=partial(self.sklearn_obj_gradient_F, A=A, tgt=tgt, C=C),
                                     hessian_f=partial(self.sklearn_obj_hessian_F, A=A, tgt=tgt, C=C),
                                     initial_p=np.ones((m, 1)), epsilon=TOL * 0.01, dom=lambda x: True)
        #self.weights = gradient_descent(f=partial(self.logistic_f, A=A, tgt=tgt),
        #                                gradient_f=partial(self.gradient_logistic_f, A=A, tgt=tgt),
        #                                initial_p=np.ones((m, 1)), epsilon=TOL, dom=lambda x: True)
        return self.weights

    def predict(self, data_entry):
        return 1 if self.logit_f(np.dot(self.weights[0].T, data_entry)) > 0.5 else 0
#class Perceptron:
#    def __init__(self):
#        pass
#
#    def train(self, A_o, tgt):
