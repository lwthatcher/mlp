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
        self.f_num = f_num
        self.c_num = c_num
        self.f_names = f_names
        self.c_names = c_names
        # get weight matrices and bias vectors
        W, b = self._weight_matrices(self._nodes_per_layer(f_num, hidden, c_num))
        self.W = W
        self.b = b

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
        if type(v) is int:
            num = v
            names = [str(i) for i in range(num)]
        else:
            num = len(v)
            names = v
        return num, names

    @staticmethod
    def _num_array(v):
        if type(v) is int:
            return [v]
        else:
            return [int(i) for i in v]

    @classmethod
    def _nodes_per_layer(cls, f, h, c):
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
