import unittest

import numpy as np

from decision_tree import DecisionTree


class DecisionTreeTests(unittest.TestCase):
    def test_nominal_feature_creates_one_child_per_value(self):
        features = np.array([[10], [20], [30], [10], [20], [30]])
        labels = np.array([0, 1, 2, 0, 1, 2])

        tree = DecisionTree(max_depth=2).fit(features, labels)

        self.assertEqual(tree.root.feature, 0)
        self.assertEqual(set(tree.root.children), {10, 20, 30})
        np.testing.assert_array_equal(tree.predict(features), labels)

    def test_unseen_value_uses_node_majority_class(self):
        features = np.array([[0], [0], [1]])
        labels = np.array([1, 1, 0])
        tree = DecisionTree().fit(features, labels)

        self.assertEqual(tree.predict(np.array([[2]]))[0], 1)

    def test_min_samples_stops_small_node(self):
        features = np.array([[0], [1], [2]])
        labels = np.array([0, 1, 1])
        tree = DecisionTree(min_samples=4).fit(features, labels)

        self.assertTrue(tree.root.is_leaf)
        self.assertEqual(tree.root.value, 1)


if __name__ == "__main__":
    unittest.main()

