import numpy as np
import argparse
import csv


class NeuralNet:

    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_nodes=20, learning_rate=0.9):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def _forward_prop(self, x):
        pass

    def _back_prop(self, y):
        pass


def load_data_file(file):
    features = []
    labels = []
    with open(file, newline='') as csv_file:
        data_reader = csv.reader(csv_file)
        for row in data_reader:
            labels.append(row[-1])
            features.append(row[:-1])
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multilayer perceptron via vector operations')
    parser.add_argument('data', help='file containing data')
    parser.add_argument('--num_hidden', '-h', type=int, help='number of hidden nodes to use')
    args = parser.parse_args()
