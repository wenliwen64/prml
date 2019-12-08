import numpy as np
from .utils import cond_check, cond_check_eq

class KMeans:
    def __init__(self, max_iters=200, epsilon=1e-6):
        self.max_iters = max_iters
        self.epsilon = epsilon

    @staticmethod
    def __group_assignment__(A, mu_init):
        """
        Assign group id to each data point.
        :param A: data, N by d numpy matrix
        :param mu: center locations, K by d numpy matrix
        :return: group membership, N by 1 vector
        """
        cond_check(len(mu_init) > 0, "number of initial group centers should be larger than 0")
        cond_check_eq(len(mu_init[0]), len(A[0]))
        ids = -1 * np.ones(shape=(len(A), 1), dtype=int)
        for id_idx, row in enumerate(A):
            lowest_dist = 0.0
            for mu_idx, mu in enumerate(mu_init):
                dist = np.linalg.norm(row - mu, ord=2)
                lowest_dist = dist if mu_idx == 0 else np.min([lowest_dist, dist])
                ids[id_idx] = mu_idx if dist <= lowest_dist else ids[id_idx]

        return ids

    @staticmethod
    def __compute_new_mu__(A, membership, K):
        """
        Have to make sure membership has K unique [0...K-1]
        :param A:
        :param membership:
        :return: new center locations
        """
        accum_center_locations = np.zeros(shape=(K, len(A[0])))
        numdata = np.zeros(shape=(K, 1))
        for group_id, row in zip(membership, A):
            accum_center_locations[group_id] += row
            numdata[group_id] += 1

        return accum_center_locations / numdata

    def train(self, A, K, seed=None):
        """
        Find the means of K groups for the given data A.
        :paramsA: n by d numpy matrix, n = # of data points, d = dimension
        :param K: number of groups
        :return: mu = the center of groups, a n by 1 vector
        """
        # randomly assign centers to K groups
        np.random.seed(seed)
        mu = A[np.random.randint(low=0, high=A.shape[0], size=K), :]
        membership = np.array([-1 for _ in range(len(A))])

        # initialize iteration difference
        mu_diff = 1.0
        iters = 0
        while mu_diff > self.epsilon and iters < self.max_iters:
            iters += 1
            membership = self.__group_assignment__(A, mu)
            new_mu = self.__compute_new_mu__(A, membership, K)
            mu_diff = np.linalg.norm([np.linalg.norm(row, ord=2) for row in (new_mu - mu)], ord=2)
            mu = new_mu

        return (mu, membership)
