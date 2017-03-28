from unittest import TestCase
from mlp import NeuralNet
from mlp.mlp import load_data_file, parse_args
import numpy as np


class TestNeuralNet(TestCase):

    def test_predict(self):
        # init with preset values
        nn = NeuralNet(2, 3, 2)
        nn.W[0] = np.array([[.6, .2, 1], [1, .5, 0]])
        nn.W[1] = np.array([[.1, 1], [1, .25], [.1, 1]])
        nn.b[0] = np.array([[0., 0., 0.]])
        nn.b[1] = np.array([[1., -1.]])
        # input values
        X = np.array([[1., .4],
                      [0., 0.],
                      [1., 1.],
                      [2., .8]])
        # expected output
        Y = np.array([[1.6, 1.1],
                      [1., 0.],
                      [1.96, 1.775],
                      [2.2, 3.2]])
        # actual output
        Z = nn.predict(X)
        # compare
        np.testing.assert_array_almost_equal(Y, Z)

    def test_forward_prop(self):
        # init with preset values
        nn = NeuralNet(2, 3, 2)
        nn.W[0] = np.array([[.1, .1, .1], [.1, .2, .3]])
        nn.W[1] = np.array([[.5, 1], [1, 2], [-2, 3]])
        nn.b[0] = np.array([[1.1, 2.1, 3.1]])
        nn.b[1] = np.array([[.3, -13]])
        # do one iteration of forward propagation
        x = np.array([1, 2])
        nn._forward_prop(x)
        # test values
        np.testing.assert_array_almost_equal(nn.Z[0], np.array([[1, 2]]))  # input vector
        np.testing.assert_array_almost_equal(nn.Z[1], np.array([[1.4, 2.6, 3.8]]))  # hidden layer output
        np.testing.assert_array_almost_equal(nn.Z[2], np.array([[0, 5]]))  # Z_out = [-4,5] ReLU(Z_out) = [0,5]

    def test_back_prop(self):
        # init with preset values
        nn = NeuralNet(2, 3, 2)
        # -weights-
        nn.W[0] = np.array([[.6, .2, 1], [1, .5, 0]])
        nn.W[1] = np.array([[.1, 1], [1, .25], [.1, 1]])
        nn.b[0] = np.array([[0., 0., 0.]])
        nn.b[1] = np.array([[0., 0.]])
        # -outputs-
        nn.Z[2] = np.array([[.6, 2.1]])
        nn.Z[1] = np.array([[1, .4, 1]])
        nn.Z[0] = np.array([[1, .4]])
        # -learning rate-
        nn.C = 1.
        # do one iteration of back-prop
        y = np.array([[1.1, 2]])
        nn._back_prop(y)
        # check that it computes the correct delta values
        np.testing.assert_array_almost_equal(nn.δ[1], np.array([[.5, -.1]]))
        np.testing.assert_array_almost_equal(nn.δ[0], np.array([[-.05, .475, -.05]]))
        # check weight updates
        np.testing.assert_array_almost_equal(nn.W[1], np.array([[.6, .9], [1.2, .21], [.6, .9]]))
        np.testing.assert_array_almost_equal(nn.W[0], np.array([[.55, .675, .95], [.98, .69, -.02]]))
        # check bias weight updates
        np.testing.assert_array_almost_equal(nn.b[1], np.array([[.5, -.1]]))
        np.testing.assert_array_almost_equal(nn.δ[0], np.array([[-.05, .475, -.05]]))

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
        self.assertEqual(len(nn.Z), 3)
        self.assertEqual(nn.Z[0].shape, (1, 5))
        self.assertEqual(nn.Z[1].shape, (1, 20))
        self.assertEqual(nn.Z[2].shape, (1, 3))
        # delta vectors
        self.assertEqual(len(nn.δ), 2)
        self.assertEqual(nn.δ[0].shape, (1, 20))
        self.assertEqual(nn.δ[1].shape, (1, 3))

    def test__num_and_names(self):
        a1, b1 = NeuralNet._num_and_names(5)
        self.assertEqual(a1, 5)
        np.testing.assert_array_equal(b1, ['0', '1', '2', '3', '4'])
        names = ['setosa', 'versicolor', 'virginica']
        a2, b2 = NeuralNet._num_and_names(names)
        self.assertEqual(a2, 3)
        np.testing.assert_array_equal(b2, names)

    def test__format_as_array(self):
        # number
        l1 = NeuralNet._format_as_array(20)
        np.testing.assert_array_equal(l1, [20])
        # array of int
        l2 = NeuralNet._format_as_array([20, 15, 30])
        np.testing.assert_array_equal(l2, [20, 15, 30])
        # array of string
        l3 = NeuralNet._format_as_array(["20", "15", "30"])
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
