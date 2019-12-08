import numpy as np
from prml.optimizer import ip_optimization, ip_hessian_f_dector, ip_f_dector, ip_grad_f_dector

class HardSVM:
    def __init__(self):
        pass

    @staticmethod
    def obj_f(w):
        return .5 * np.linalg.norm(w[:-1], ord=2) ** 2.0

    @staticmethod
    def gradient_f(w):
        return np.append(w[:-1], 0)

    @staticmethod
    def hessian_f(w):
        res = np.eye(len(w))
        res[-1, -1] = 0.0
        return res

    #@staticmethod
    #def hessian_f(w):
    #    ww = np.append(w[:-1], 0)
    #    hessian_f_matrix = np.zeros(shape=(len(ww), len(ww)))
    #    for row_i in range(len(ww)):
    #        for col_j in range(len(ww)):
    #            if row_i == col_j:
    #                hessian_f_matrix[row_i, col_j] += 2.
    #            else:
    #                hessian_f_matrix[row_i, col_j] += 2. * ww[row_i] + 2. * ww[col_j]

    #    gradient_f_vector = HardSVM.gradient_f(ww).reshape(len(ww), 1)
    #    return -np.linalg.norm(ww, ord=2) ** (-3.0) * hessian_f_matrix \
    #           + 1.5 * np.linalg.norm(ww, ord=2) ** (-5.0) * np.matmul(gradient_f_vector, gradient_f_vector.T)

    def train(self, X, y):
        b = -1. * np.ones(shape=(len(y),))
        # (w, intercept) * (x, 1) < -1
        # (w, intercept) * (-x, -1) < -1
        A = []
        for x, sample_y in zip(X, y):
            xx = np.append(x, 1.)
            if sample_y == 1:
                A.append(-xx)
            elif sample_y == 0:
                A.append(xx)
        A = np.array(A)

        # find the feasible initial point
        print('A.shape', A.shape)
        feasible_A = np.append(A, -1. * np.ones(shape=(A.shape[0],1)), axis=1)
        feasible_b = np.zeros(shape=(A.shape[0],))
        def feasible_obj_f(x):
            return x[-1]

        def feasible_grad_f(x):
            res = np.zeros(shape=(len(x),))
            res[-1] = 1.0
            return res

        def feasible_hessian_f(x):
            return np.zeros(shape=(len(x), len(x)))

        feasible_initial_p = np.zeros(shape=(feasible_A.shape[1],))
        feasible_initial_p[-1] = 1.0

        feasible_initial_p = ip_optimization(feasible_obj_f, feasible_grad_f, feasible_hessian_f,
                                              initial_p=feasible_initial_p, A=feasible_A, b=feasible_b)
        feasible_initial_p = -feasible_initial_p / feasible_initial_p[-1] * 1.1
        print('initial_p', feasible_initial_p)
        print('constraints:', np.dot(A, feasible_initial_p[:-1]))
        print('------>feasible_initial_p', feasible_initial_p)
        print('\n\n\n\n==================Start Optimization SVM==========================\n\n\n\n')

        x = ip_optimization(HardSVM.obj_f, HardSVM.gradient_f, HardSVM.hessian_f,
                            initial_p=feasible_initial_p[:-1], A=A, b=b)
        print('------>', x)
        return x


