import numpy as np
import argparse
import csv
import sys


class NeuralNet:

    def __init__(self, features, classes, hidden_layers=20, learning_rate=0.9):
        """
        Initializes the neural network with the
        :type features: int or array-of-string
        :type classes: int or array-of-string
        :param features: The input features to expect.
            int: the number of features to expect
            array-of-string: the names of the features to expect
        :param classes: The valid output classifications.
        :param hidden_layers:
        :param learning_rate:
        """
        pass

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
