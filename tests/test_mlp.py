from unittest import TestCase
from mlp import NeuralNet
from mlp.mlp import load_data_file, parse_args
import numpy as np


class TestNeuralNet(TestCase):
    def test__constructor(self):
        nn = NeuralNet(5, 20, 3)
        # weight matrices
        self.assertEqual(len(nn.W), 2)
        self.assertEqual(nn.W[0].shape, (5, 20))
        self.assertEqual(nn.W[1].shape, (20, 3))
        # bias weights
        self.assertEqual(len(nn.b), 2)
        self.assertEqual(nn.b[0].shape, (1, 20))
        self.assertEqual(nn.b[1].shape, (1, 3))
        # output vectors
        self.assertEqual(len(nn._Z), 3)
        self.assertEqual(nn._Z[0].shape, (1, 5))
        self.assertEqual(nn._Z[1].shape, (1, 20))
        self.assertEqual(nn._Z[2].shape, (1, 3))

    def test__num_and_names(self):
        a1, b1 = NeuralNet._num_and_names(5)
        self.assertEqual(a1, 5)
        np.testing.assert_array_equal(b1, ['0', '1', '2', '3', '4'])
        names = ['setosa', 'versicolor', 'virginica']
        a2, b2 = NeuralNet._num_and_names(names)
        self.assertEqual(a2, 3)
        np.testing.assert_array_equal(b2, names)

    def test__num_array(self):
        # number
        l1 = NeuralNet._num_array(20)
        np.testing.assert_array_equal(l1, [20])
        # array of int
        l2 = NeuralNet._num_array([20, 15, 30])
        np.testing.assert_array_equal(l2, [20, 15, 30])
        # array of string
        l3 = NeuralNet._num_array(["20", "15", "30"])
        np.testing.assert_array_equal(l3, [20, 15, 30])

    def test__nodes_per_layer(self):
        l1 = NeuralNet._nodes_per_layer(5, 20, 3)
        np.testing.assert_array_equal(l1, [5, 20, 3])
        l2 = NeuralNet._nodes_per_layer(5, [20, 13], 8)
        np.testing.assert_array_equal(l2, [5, 20, 13, 8])

    def test__weight_matrices(self):
        W, b = NeuralNet._weight_matrices([5, 20, 3])
        self.assertEqual(len(W), 2)
        self.assertEqual(len(b), 2)
        self.assertEqual(W[0].shape, (5, 20))
        self.assertEqual(W[1].shape, (20, 3))
        self.assertEqual(b[0].shape, (1, 20))
        self.assertEqual(b[1].shape, (1, 3))


class TestCLI(TestCase):
    def test_load_data_file(self):
        iris = "../examples/iris.txt"
        features, labels = load_data_file(iris)
        self.assertEqual(features.shape, (150, 4))
        self.assertEqual(labels.shape, (150,))
        np.testing.assert_array_equal(features[99, :], np.array([5.7, 2.8, 4.1, 1.3]))
        self.assertEqual(labels[99], 1.)
