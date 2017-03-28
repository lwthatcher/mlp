import numpy as np
import argparse
import csv
from mlp.activation_functions import ReLU, Sigmoid
from mlp import util


class NeuralNet:

    def __init__(self, features, hidden=20, classes=2, learning_rate=0.9, a_func=ReLU, max_epochs=1000, patience=20,
                 validation_set=None):
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
        :param a_func: The activation function to use. Default = ReLU
        :param max_epochs: maximum number of epochs to train for
        :param patience: number of iterations to check for accuracy improvement
        :param validation_set: tuple (Vx, Vy)
            Vx: the validation training data
            Vy: the validation test labels
        """
        f_num, f_names = util.num_and_names(features)
        c_num, c_names = util.num_and_names(classes)
        # save names of input/output features
        self._features = f_names
        self._output_classes = c_names
        # get weight matrices and bias weight vectors
        layers = self._nodes_per_layer(f_num, hidden, c_num)
        W, b = self._weight_matrices(layers)
        self.W = W
        self.b = b
        # setup output and delta vectors
        self.Z = [np.zeros((1, l)) for l in layers]
        self.δ = [np.zeros((1, l)) for l in layers[1:]]
        # learning rate
        self.C = learning_rate
        # Activation Function
        self.activation = a_func.activation
        self.f_prime = a_func.f_prime
        # helper variables
        self.num_layers = len(layers)
        # stopping criteria
        if max_epochs is None:
            max_epochs = np.inf
        self._max_epochs = max_epochs
        self._patience = patience
        self._VS = validation_set

    def fit(self, X, Y):
        epoch = 0
        num_samples = len(X)
        while epoch < self._max_epochs:
            idx = util.shuffle_indices(num_samples)
            for i in idx:
                self._forward_prop(X[i])
                self._back_prop(Y[i])
            # TODO: implement BSSF, validation set, accuracy checks, and patience stopping criteria

    def predict(self, X):
        out = []
        for x in X:
            out.append(self._forward_prop(x))
        # TODO: for classification take argmax
        return np.array(out)

    def _forward_prop(self, x):
        self.Z[0] = x.reshape(1, len(x))
        for i in range(self.num_layers-1):
            self.Z[i + 1] = self.activation(self.Z[i].dot(self.W[i]) + self.b[i])
        return self.Z[-1][0]

    def _back_prop(self, y):
        # output layer's delta: δ = (T-Z) * f'(net)
        self.δ[-1] = (y - self.Z[-1]) * self.f_prime(self.Z[-1])
        # compute deltas: δj = Σ[δk*Wjk] * f'(net)
        for i in range(self.num_layers-2, 0, -1):
                self.δ[i-1] = np.tensordot(self.δ[i], self.W[i], (1, 1)) \
                              * self.f_prime(self.Z[i])
        # update weights: ΔWij = C*δj*Zi
        for i in range(self.num_layers-2, -1, -1):
            # Note since δ,W,b are all of length: num_layers-1, layer(Z[i]) == layer(b[i+1])
            self.W[i] += self.C * np.outer(self.Z[i], self.δ[i])
            self.b[i] += self.C * self.δ[i]

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
        layers.extend(util.format_as_array(h))
        layers.append(c)
        return layers

    @staticmethod
    def _weight_matrices(layers):
        """
        Creates the weights matrices (W) and the bias vectors (b) for each appropriate layer
        :param layers: array of integers with number of nodes per layer
        :return: W, b
        """
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
