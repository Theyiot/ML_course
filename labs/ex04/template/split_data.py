# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    data_size = x.shape[0]
    training_indices = np.random.choice(data_size, int(data_size * ratio), replace=False)
    test_indices = np.delete(np.array(range(data_size)), training_indices) 
    return x[training_indices], y[training_indices], x[test_indices], y[test_indices]
