#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import *


def ConstructWbyEdge(X, E, t=1):
    """


    """

    dataNum = np.size(X, axis=0)
    D = pairwise_distances(X)
    D_ = np.multiply(D, E)
    W_ = np.exp(-D_/(2*t*t))
    for i in range(dataNum):
        for j in range(dataNum):
            if W_[i, j] == 1:
                W_[i, j] = 0
    W = csc_matrix(W_)
    return W
