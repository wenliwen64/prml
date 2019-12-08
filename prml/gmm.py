import numpy as np
from scipy.stats import multivariate_normal as gaussian
#from .utils import gaussian

class GMM:
    def __init__(self, n_groups, random_seed):
        """
        Initialize parameter vectors
        :param n_groups:
        """
        #self.pi = []  # k pi's
        #self.mu = []  # k mu's
        #self.Sigma = []  # k covariance matrices
        self.n_groups = n_groups
        self.rng = np.random.RandomState(random_seed)
        self.epsilon = 1e-6 # todo
        self.max_iters = 200 # todo

    def _rnd_init_pi_(self):
        """
        Random initialization of
        :param random_seed:
        :return:
        """
        return 1./ self.n_groups * np.ones(shape=(self.n_groups, 1))


    def _rnd_init_mu_(self, X):
        """

        :param X:
        :return:
        """
        return X[self.rng.randint(low=0, high=len(X), size=self.n_groups), :]


    def _init_Sigma_(self, X):
        """

        :param X: n by d
        :return: d by d covariance matrix
        """
        return np.array([np.eye(X.shape[1]) for _ in range(self.n_groups)])


    def train(self, X):
        """
        EM algorithm estimating parameters (pi_{1:k}, mu_{1:k}, Sigma_{1:k})
        :param X: n by d numpy matrix of data
        :return:
        """
        n_samples = len(X)
        # Random initialization
        pi = self._rnd_init_pi_()
        mu = self._rnd_init_mu_(X)
        Sigma = self._init_Sigma_(X)

        (q_diff, iters) = (1.0, 0)
        old_qz = np.zeros(shape=(n_samples, self.n_groups))

        while q_diff > self.epsilon and iters < self.max_iters:
            iters += 1
            if (iters % 10 == 0):
                print(iters, "=========", q_diff)

            # E-step q(z=j|x_i) = p(x_i|z=j)p(z=j) / Norm
            qz = np.zeros(shape=(n_samples, self.n_groups))
            for i, X_row in enumerate(X):
                norm = np.sum([gaussian(mean=mu[j], cov=Sigma[j]).pdf(X_row) * pi_j for j, pi_j in enumerate(pi)])
                qz[i, :] = np.array([gaussian(mean=mu[j], cov=Sigma[j]).pdf(X_row) * pi_j / norm for j, pi_j in enumerate(pi)]).flat

            # M-step
            for idx_pi in range(self.n_groups):
                pi[idx_pi] = 1. / n_samples * np.sum(qz[:, idx_pi])
                mu[idx_pi] = np.matmul(X.transpose(), qz[:, idx_pi]).transpose() / np.sum(qz[:, idx_pi])
                Sigma[idx_pi] = np.sum([np.matmul((X_row - mu[[idx_pi]]).transpose(), X_row - mu[[idx_pi]]) * qz[idx_X, idx_pi] for idx_X, X_row in enumerate(X)], axis=0) \
                                / np.sum(qz[:, idx_pi])

            q_diff = np.linalg.norm(qz - old_qz, ord=1)
            old_qz = qz

        return (pi, mu, Sigma, np.argmax(qz, axis=1))
