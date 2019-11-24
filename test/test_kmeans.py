from prml.kmeans import KMeans as km
import numpy as np
from sklearn.cluster import KMeans as sk_km
import pandas as pd


class TestKM:
    def test_group_assignment(self):
        A = np.array([
            [1, 1],
            [5, 5],
            [.5, .5]
        ])

        mu_init = np.array([
            [1.5, 1.5],
            [4.0, 4.5]
        ])

        ids = km.__group_assignment__(A, mu_init)
        assert ids[0] == 0
        assert ids[1] == 1
        assert ids[2] == 0

    def test_compute_new_mu__(self):
        A = np.array([
            [1.0, 1.0],
            [5.0, 5.0],
            [0.5, 0.5],
            [3.0, 3.0],
            [4.0, 4.0],
            [4.5, 4.5],
            [4.85, 5.0]
        ])

        membership = np.array([
           [1], [1],
           [2], [2],
           [0], [0], [0]
        ])

        new_mu = km.__compute_new_mu__(A, membership, 3)
        new_mu_expected = np.array([
            [4.45, 4.5],
            [3.0, 3.0],
            [1.75, 1.75]
        ])
        for mu, mu_expected in zip(new_mu_expected, new_mu_expected):
            assert np.array_equal(mu, mu_expected)


    def test_train(self):
        sk_kmeans = sk_km(n_clusters=2, init='random', random_state=1)
        my_kmeans = km()
        geyser_data = pd.read_csv("../data/faithful.csv", index_col=0)
        sk_res = sk_kmeans.fit(geyser_data)
        my_res = my_kmeans.train(geyser_data.values, K=2, seed=10)
        assert np.array_equal(my_res[1], sk_res.labels_.reshape((len(geyser_data), 1)))
