from unittest import TestCase
import numpy as np
from mlp.activation_functions import ReLU


class TestReLU(TestCase):
    def test_activation(self):
        x = np.array([[-4, 5]])
        ReLU.activation(x)
        np.testing.assert_array_equal(x, np.array([[0, 5]]))

    def test_f_prime(self):
        x = np.array([[-4, 5]])
        ReLU.f_prime(x)
        np.testing.assert_array_equal(x, np.array([[0, 1]]))
