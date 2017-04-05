import numpy as np
from mlp import NeuralNet
from mlp import util
from mlp.activation_functions import ReLU, Sigmoid


def run_blobs(n=5):
    X, Y = get_blob_sets(n)
    net = NeuralNet(4, 20, 6, a_func=Sigmoid, validation_set=(X[2], Y[2]))
    num_epochs = net.fit(X[:2], Y[:2], True)
    score = net.score(X[-1], Y[-1])
    print("accuracy:", score)
    print("epochs:", num_epochs)


def get_blob_sets(n):
    X = []
    Y = []
    for i in range(n):
        x, lbls = util.load_data_file("blobs_" + str(i) + ".txt")
        y = util.to_output_vector(lbls)
        X.append(x)
        Y.append(y)
    return X, Y

if __name__ == '__main__':
    run_blobs()
