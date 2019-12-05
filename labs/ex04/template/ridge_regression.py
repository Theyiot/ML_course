# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    A = tx.T.dot(tx)
    I = np.identity(A.shape[0])
    w = np.linalg.solve(A + lambda_ * 2 * len(y) * I, tx.T.dot(y))
    return compute_mse(y, tx, w), w