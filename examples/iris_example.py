import numpy as np
from mlp import NeuralNet
from mlp import util
from sklearn.preprocessing import normalize
from mlp.activation_functions import ReLU, Sigmoid
from sklearn.model_selection import cross_val_score


def run_iris():
    features, labels = util.load_data_file("iris.txt")
    features = normalize(features, axis=0)
    # get validation set
    # num_examples = len(labels)
    # idx = np.arange(num_examples)
    # v_idx = np.random.choice(idx, 10, replace=False)
    # val_x = features[v_idx]
    # val_y = util.to_output_vector(labels[v_idx])
    # train/test set
    X = features
    Y = util.to_output_vector(labels)
    args = [4, 10, 3]
    kwargs = {"max_epochs": 1000, "a_func": Sigmoid}
    cross_fold(X, Y, 10, *args, **kwargs)


def cross_fold(X, Y, n, *model_args, **model_kwargs):
    num_examples = len(Y)
    # shuffle data first
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    # split into n sets
    splits = np.split(idx, n)
    for i in range(n):
        # get train/test sets
        idx_test = splits[i]
        tr1 = splits[:i]
        tr2 = splits[i+1:]
        tr1.extend(tr2)
        idx_train = np.concatenate(tr1)
        X_train = X[idx_train,:]
        X_test = X[idx_test,:]
        Y_train = Y[idx_train]
        Y_test = Y[idx_test]
        # create new model
        model = NeuralNet(*model_args, **model_kwargs)
        # train
        num_epochs = model.fit(X_train, Y_train)
        # compare
        out = model.predict(X_test)
        print(i, model.score(X_train, Y_train), num_epochs)


if __name__ == '__main__':
    run_iris()
