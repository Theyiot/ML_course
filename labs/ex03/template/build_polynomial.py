# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = []
    for x_i in x:
        poly.append(np.power(np.repeat(x_i, degree + 1), range(degree + 1)))
    return np.array(poly)