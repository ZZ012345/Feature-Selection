#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

def ConstructAffinityMatrix(X):
    dataNum, dataDim = X.shape
    A = np.sum(np.square(X), axis=1)
    B = np.tile(A, (dataNum, 1))
    C = B + B.T - 2 * np.dot(X, X.T)
    for i in range(dataNum):
        C[i, i] = 0
    return C