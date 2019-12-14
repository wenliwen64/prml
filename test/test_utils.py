from prml.utils import (cond_check, cond_check_eq, gini, cross_entropy, partition_equal)
import pytest
import numpy as np

class TestCondCheck:
    def test_cond_check(self):
        with pytest.raises(AssertionError, match="7 is not less than 5"):
            (a, b) = (7, 5)
            cond_check(a < b, a, "is not less than", b)

    def test_cond_check_eq(self):
        with pytest.raises(AssertionError, match="5 is not equal to 6"):
            (a, b) = (5, 6)
            cond_check_eq(a, b)


def test_cross_entropy():
    assert np.isclose(cross_entropy([1, 1, 1, 0, 0]), 0.67301166)
    assert np.isclose(cross_entropy([1, 2, 3]), -np.log(1 / 3))
    assert np.isclose(cross_entropy([1]), 0.0)
    assert np.isclose(cross_entropy([]), 0.0)


def test_gini():
    assert np.isclose(gini([1, 1, 1, 0, 0]), .48)
    assert np.isclose(gini([1, 2, 3]), 2. / 3.)
    assert np.isclose(gini([1]), 0.)
    assert np.isclose(gini([]), 0.)


class TestPartitionCheck:
    def test_shape(self):
        assert not partition_equal([0,1,1,2], [0,1,1,2,2])
        assert partition_equal([], [])

    def test_id_sets(self):
        assert not partition_equal([0,1,2,3], [1,1,3,4])
        assert partition_equal([0,1,2], [2,1,0])

    def test_partion_equal(self):
        assert partition_equal([0, 1, 2, 3], [1, 2, 3, 5])
        assert not partition_equal([0, 0, 1, 2, 3], [1, 2, 2, 3, 4])
