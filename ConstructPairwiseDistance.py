#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


def ConstructPairwiseDistance(X, version=0):
    dataNum, dataDim = X.shape
    if version == 0:
        A = np.sum(np.square(X), axis=1)
        B = np.tile(A, (dataNum, 1))
        C = B + B.T - 2 * np.dot(X, X.T)
        for i in range(dataNum):
            C[i, i] = 0
        return np.sqrt(C)
    else:
        W = np.zeros((dataNum, dataNum))
        for i in range(dataNum-1):
            diffMat = np.tile(X[i, :], (dataNum-i-1, 1)) - X[i+1:dataNum, :]
            squareDiffMat = np.square(diffMat)
            sumDistMat = np.sum(squareDiffMat, axis=1)
            distMat = np.sqrt(sumDistMat)
            W[i, i+1:dataNum] = distMat
            W[i+1:dataNum, i] = distMat.T
        return W
