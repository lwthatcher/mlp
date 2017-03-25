from unittest import TestCase
import numpy as np
from mlp.activation_functions import ReLU, Sigmoid


class TestReLU(TestCase):
    def test_activation(self):
        x = np.array([[-4, 5]])
        ReLU.activation(x)
        np.testing.assert_array_equal(x, np.array([[0, 5]]))
        a = np.array([[-2, 5]])
        b = np.array([[-2, 5]])
        y = ReLU.activation(a + b)
        np.testing.assert_array_equal(y, np.array([[0, 10]]))
        np.testing.assert_array_equal(a, np.array([[-2, 5]]))
        np.testing.assert_array_equal(b, np.array([[-2, 5]]))

    def test_f_prime(self):
        x = np.array([[-4, 5]])
        ReLU.f_prime(x)
        np.testing.assert_array_equal(x, np.array([[0, 1]]))


class TestSigmoid(TestCase):
    def test_activation(self):
        x = np.array([[-.8, .62005]])
        Sigmoid.activation(x)
        np.testing.assert_array_almost_equal(x, np.array([[.310026, .65023]]))
        a = np.array([[-.4, .62]])
        b = np.array([[-.4, .00005]])
        y = Sigmoid.activation(a + b)
        np.testing.assert_array_almost_equal(y, np.array([[.310026, .65023]]))
        np.testing.assert_array_equal(a, np.array([[-.4, .62]]))
        np.testing.assert_array_equal(b, np.array([[-.4, .00005]]))

    def test_f_prime(self):
        x = np.array([[.1, .3, .5, 1]])
        Sigmoid.f_prime(x)
        np.testing.assert_array_almost_equal(x, np.array([[.09, .21, .25, 0]]))
