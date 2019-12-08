import numpy as np
from prml.optimizer import (ip_f_dector, ip_grad_f_dector, ip_hessian_f_dector, ip_optimization)
from prml.svm import HardSVM


class TestDectors:
    """
    min x1^2 + x2^2 + 2x1x2
    s.t. -x1 - x2 < -3.0
          x1 - x2 < -2.0

    so Ax < b
       A = [[-1., -1.],
            [ 1., -1.]]
       b = [[-3.], [-2.]]
    """
    quad_matrix = np.array([[1., 1.], [1., 1.]])
    constraint_m = np.array([[-1., -1.],
                             [1., -1.]])
    constraint_b = np.array([-3., -2.])

    def test_ip_dector(self):

        def obj_f(x):
            return np.dot(x, np.dot(TestDectors.quad_matrix, x))
        assert obj_f(np.array([1.0, 4.0])) == 25.0

        new_obj_f = ip_f_dector(f=obj_f, t=2.0, A=TestDectors.constraint_m,
                                b=TestDectors.constraint_b)
        assert new_obj_f(np.array([1.0, 4.0])) == 50.0 - np.log(2.0)

    def test_ip_grad_f_dector(self):

        def grad_f(x):
            return np.array(2 * np.dot(TestDectors.quad_matrix, x))

        assert np.array_equal(grad_f(np.array([1.0, 4.0])), np.array([10.0, 10.0]))

        new_grad_f = ip_grad_f_dector(grad_f=grad_f, t=2.0, A=TestDectors.constraint_m,
                                      b=TestDectors.constraint_b)

        assert np.array_equal(new_grad_f(np.array([1.0, 4.0])), np.array([20.5, 18.5]))

    def test_ip_hessian_f_dector(self):

        def hessian_f(x):
            return np.array([[2., 0.], [0., 2.0]])

        new_hessian_f = ip_hessian_f_dector(hessian_f=hessian_f, t=2.0,
                                            A=TestDectors.constraint_m,
                                            b=TestDectors.constraint_b)

        assert np.array_equal(new_hessian_f(np.array([1.0, 4.0])),
                                            np.array([[5.25, -.75], [-.75, 5.25]]))

    def test_train(self):
        from sklearn.datasets.samples_generator import make_blobs
        X, y = make_blobs(n_samples=50, centers=2,
                          random_state=0, cluster_std=0.60)
        svc = HardSVM()
        w = svc.train(X, y)
        assert np.linalg.norm(w - [0.23525694, -1.41250783, 3.29634152], ord=2) < 1.e-3


    def test_ip_optimization(self):

        def obj_f(x):
            return x[2]

        def grad_f(x):
            return np.array([0, 0, 1])

        def hessian_f(x):
            return np.zeros((3, 3))

        A = np.array([[-1, -1, 0],
                      [1, -1, 0],
                      [0.1, 1, -1]])
        b = np.array([-4000.0, 4000.0, -0.0])
        x = ip_optimization(obj_f, grad_f, hessian_f,
                              initial_p=[9000.0, 10000.0, 20000.0],
                              A=A,
                              b=b)
        assert np.abs(x[2] / 400.0 - 1.) < 1.e-6


