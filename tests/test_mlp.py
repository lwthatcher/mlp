from unittest import TestCase
from mlp import NeuralNet
from mlp.mlp import load_data_file, parse_args
import numpy as np


class TestNeuralNet(TestCase):
    def test__num_and_names(self):
        self.fail()


class TestCLI(TestCase):
    def test_load_data_file(self):
        iris = "../examples/iris.txt"
        features, labels = load_data_file(iris)
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))
        np.testing.assert_array_equal(features[99, :], np.array([5.7, 2.8, 4.1, 1.3]))
        self.assertEqual(labels[99], 1.)
