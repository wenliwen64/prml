from prml import tree, utils
import numpy as np

def test_find_optimal_cut_point():
    X_y = np.array([
        [1.0, 0], [2.0, 0], [3.0, 0],
        [4.0, 1], [5.0, 1],
        [6.0, 2], [7.0, 2], [8.0, 2], [9.0, 2]
    ])

class TestNode:
    def test_majority(self):
        node = tree.DecisionTreeNode(observations=np.random.rand(4, 2), tgts=np.array([0, 0, 1, 2]))
        assert node.majority == 0
        assert node.total_samples_num == 4
        assert node.num_classes == 3

class TestTree:
    def test_train(self):
        X_y = np.array([
            [1.0, 1.5, 0], [.5,   1., 0],
            [4.2,  .8, 1], [5.5, 1.5, 1],
            [3.5, 2.3, 2], [5.7, 9.2, 2],
            [6.4, 2.1, 3], [9.0, 3.2, 3]
        ])
        dtree = tree.DecisionTree()
        dtree.train(X_y[:, :-1], X_y[:, -1])
        assert np.array_equal(dtree.root.predict(np.array([
             [2.5, 1.7] #0
            ,[4.5, .8]  #1
            ,[6.4, 3.4] #3
            ,[5.5, 2.9] #2
        ])), [0, 1, 3, 2])
        #print("prediction", dtree.root.predict(np.array([
        #     [2.5, 1.7] #0
        #    ,[4.5, .8]  #1
        #    ,[6.4, 3.4] #3
        #    ,[5.5, 2.9] #2
        #])))
