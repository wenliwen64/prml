from prml.gmm import GMM
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from prml.utils import partition_equal

N_GROUPS=4
N_SAMPLES=400


@pytest.fixture(scope='module')
def import_stretched_gmm_data():
    from sklearn.datasets.samples_generator import make_blobs
    X, y_true = make_blobs(n_samples=N_SAMPLES, centers=N_GROUPS,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting

    rng = np.random.RandomState(13)
    X_stretched = np.dot(X, rng.randn(2, 2))
    return (X, X_stretched, y_true)


class TestGMM:
    def test_gmm(self, import_stretched_gmm_data):
        (X, X_stretched, y_true) = import_stretched_gmm_data

        sk_gmm = GaussianMixture(n_components=N_GROUPS)
        sk_gmm.fit(X_stretched)

        gmm = GMM(n_groups=N_GROUPS, random_seed=10)
        (pi, mu, Sigma, labels) = gmm.train(X_stretched)

        assert partition_equal(sk_gmm.predict(X_stretched), labels)

