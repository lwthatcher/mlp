import numpy as np
import argparse
import csv
import sys


class NeuralNet:

    def __init__(self, features, classes, hidden=20, learning_rate=0.9):
        """
        Initializes the neural network.
        :type features: int or array-of-string
        :type classes: int or array-of-string
        :type hidden: int or array-of-int
        :param features: The input features to expect.
            int: the number of features to expect
            array-of-string: the names of the features to expect
        :param classes: The valid output classifications.
        :param hidden: Specifies the number of nodes for hidden layer(s).
            int: one hidden layer with n nodes, where hidden=n
            array-of-int: m hidden layers, where layer i in [0..m] have hidden[i] number of nodes
        :param learning_rate: The learning rate to scale results by.
        """
        f_num, f_names = self._num_and_names(features)
        c_num, c_names = self._num_and_names(classes)
        # save names of input/output features
        self._features = f_names
        self._output_classes = c_names
        # get weight matrices and bias vectors
        layers = self._nodes_per_layer(f_num, hidden, c_num)
        W, b = self._weight_matrices(layers)
        self.W = W
        self.b = b
        # setup output values
        self._Z = []
        for l in layers[1:]:
            self._Z.append(np.zeros(l))
        # learning rate
        self.C = learning_rate

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def _forward_prop(self, x):
        pass

    def _back_prop(self, y):
        pass

    @staticmethod
    def _num_and_names(v):
        """
        Converts the input v into both the length (num),
        and a list of names (names)
        :param v: int or array-of-string
            int: The length. Names will be the numbers.
            array-of-string: The names. Num will be the length of v.
        """
        if type(v) is int:
            num = v
            names = [str(i) for i in range(num)]
        else:
            num = len(v)
            names = v
        return num, names

    @staticmethod
    def _num_array(v):
        """
        Makes sure the provided argument is in array form, not just an int.
        """
        if type(v) is int:
            return [v]
        else:
            return [int(i) for i in v]

    @classmethod
    def _nodes_per_layer(cls, f, h, c):
        """
        Gets the number of nodes for each layer,
        where the first layer is the input layer
        and the last layer is the output layer
        :param f: The number of nodes for the input layer (# of features)
        :param h: int or array-of-int
            The number of nodes for the hidden layers
        :param c: The number of nodes for the output layer (# of output classifications)
        :return: array-of-int
            The number of nodes for each layer
        """
        layers = [f]
        layers.extend(cls._num_array(h))
        layers.append(c)
        return layers

    @staticmethod
    def _weight_matrices(layers):
        W = []
        b = []
        for i in range(len(layers)-1):
            W.append(np.random.randn(layers[i], layers[i+1]))
            b.append(np.random.randn(1, layers[i+1]))
        return W, b


def load_data_file(file):
    features = []
    labels = []
    with open(file, newline='') as csv_file:
        data_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            labels.append(row[-1])
            features.append(row[:-1])
    return np.array(features), np.array(labels)


def parse_args(_args=None):
    parser = argparse.ArgumentParser(description='Run a multilayer perceptron via vector operations')
    parser.add_argument('data', help='file containing data')
    parser.add_argument('--num_hidden', '-H', type=int, help='number of hidden nodes to use')
    if _args is None:
        return parser.parse_args()
    return parser.parse_args(_args)

if __name__ == '__main__':
    args = parse_args()
    print(args)
