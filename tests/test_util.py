from unittest import TestCase
from mlp import util
import numpy as np


class TestUtil(TestCase):

    def test_format_as_array(self):
        # number
        l1 = util.format_as_array(20)
        np.testing.assert_array_equal(l1, [20])
        # array of int
        l2 = util.format_as_array([20, 15, 30])
        np.testing.assert_array_equal(l2, [20, 15, 30])
        # array of string
        l3 = util.format_as_array(["20", "15", "30"])
        np.testing.assert_array_equal(l3, [20, 15, 30])

    def test_num_and_names(self):
        a1, b1 = util.num_and_names(5)
        self.assertEqual(a1, 5)
        np.testing.assert_array_equal(b1, ['0', '1', '2', '3', '4'])
        names = ['setosa', 'versicolor', 'virginica']
        a2, b2 = util.num_and_names(names)
        self.assertEqual(a2, 3)
        np.testing.assert_array_equal(b2, names)

    def test_load_data_file(self):
        iris = "../examples/iris.txt"
        features, labels = util.load_data_file(iris)
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))
        np.testing.assert_array_equal(features[99, :], np.array([5.7, 2.8, 4.1, 1.3]))
        self.assertEqual(labels[99], 1.)