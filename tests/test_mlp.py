from unittest import TestCase
from mlp import NeuralNet
from mlp.mlp import load_data_file, parse_args
import numpy as np


class TestNeuralNet(TestCase):
    def test__num_and_names(self):
        a1, b1 = NeuralNet._num_and_names(5)
        self.assertEqual(a1, 5)
        np.testing.assert_array_equal(b1, ['0', '1', '2', '3', '4'])
        names = ['setosa', 'versicolor', 'virginica']
        a2, b2 = NeuralNet._num_and_names(names)
        self.assertEqual(a2, 3)
        np.testing.assert_array_equal(b2, names)


class TestCLI(TestCase):
    def test_load_data_file(self):
        iris = "../examples/iris.txt"
        features, labels = load_data_file(iris)
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))
        np.testing.assert_array_equal(features[99, :], np.array([5.7, 2.8, 4.1, 1.3]))
        self.assertEqual(labels[99], 1.)
