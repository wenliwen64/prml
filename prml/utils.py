import math
import numpy as np
from collections import Counter, defaultdict


def cond_check(*args):
    assert args[0], ' '.join(map(str,args[1:]))


def cond_check_eq(*args):
    msg = ' '.join([str(args[0]), "is not equal to", str(args[1])])
    assert args[0] == args[1], msg


def gaussian(x, mu, Sigma):
    """

    :param Sigma:  n by n
    :return: scaler value
    """
    return ((2. * math.pi)**len(mu) * np.linalg.det(Sigma))**-.5 \
         * np.exp(-.5 * (x - mu).dot(np.linalg.inv(Sigma)).dot(x - mu))


def node_score(score_f):
    def new_score_f(tgts):
        if len(tgts) == 0:
            return 0.
        tgt_cnter = Counter(tgts)
        return np.sum([score_f(cnts / len(tgts)) for cnts in tgt_cnter.values()])

    return new_score_f

@node_score
def gini(p):
    return p * (1 - p)


@node_score
def cross_entropy(p):
    return -p * np.log(p)


def partition_equal(lhs, rhs):
    """
    To check if two partitions are equal
    :param lhs: n by 1 vector including index from 0 to N - 1
    :param rhs: n by 1 vecgor including index from 0 to N - 1
    :return:
    """
    # check shape
    if np.shape(lhs) != np.shape(rhs):
        return False

    lhs_ids, rhs_ids = np.unique(lhs), np.unique(rhs)

    # check id sets are the same
    if len(lhs_ids) != len(rhs_ids):
        return False

    lhs_gid2idx, rhs_gid2idx = defaultdict(list), defaultdict(list)

    for (idx_lhs, gid_lhs), (idx_rhs, gid_rhs) in zip(enumerate(lhs), enumerate(rhs)):
        lhs_gid2idx[gid_lhs].append(idx_lhs)
        rhs_gid2idx[gid_rhs].append(idx_rhs)

    # re-id lists using their first number
    (lhs_gid, rhs_gid) = ({sorted(l)[0]: sorted(l) for _, l in lhs_gid2idx.items()},
                          {sorted(l)[0]: sorted(l) for _, l in rhs_gid2idx.items()})

    return lhs_gid == rhs_gid
