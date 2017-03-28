"""Module for holding common utility and helper functions"""
import numpy as np


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
