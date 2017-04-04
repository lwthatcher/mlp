"""Module for holding common utility and helper functions"""
import numpy as np
import csv
from sklearn import preprocessing


def format_as_array(v):
    if type(v) is int:
        return [v]
    else:
        return [int(i) for i in v]


def shuffle_indices(num_samples):
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    return idx


def num_and_names(v):
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


def load_data_file(file):
    features = []
    labels = []
    with open(file, newline='') as csv_file:
        data_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            labels.append(row[-1])
            features.append(row[:-1])
    return np.array(features), np.array(labels)


def to_output_vector(labels):
    f = preprocessing.LabelBinarizer()
    return f.fit_transform(labels)


class BSSF:
    def __init__(self, W, b, score):
        self.W = [Wi.copy() for Wi in W]
        self.b = [bi.copy() for bi in b]
        self.score = score
