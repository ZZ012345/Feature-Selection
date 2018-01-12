#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy.io
from ReadFile import ReadFile
import construct_W
import datetime
from ConstructPairwiseDistance import ConstructPairwiseDistance
import ConstructWbyEdge
from scipy.sparse import *


X = np.array([[0, 0], [1, 1], [2, 2], [4, 4]])
E = np.array([[0, 1, 0, 0], [1,0,1,0],[0,1,0,1],[0,0,1,0]])
print X
print E
D = pairwise_distances(X)
print D
D_ = np.multiply(D, E)
print D_
W = np.exp(-D_/2)
for i in range(4):
    for j in range(4):
        if W[i, j] == 1:
            W[i, j] = 0
print W
sparseW = csc_matrix(W)
print sparseW
print ConstructWbyEdge.ConstructWbyEdge(X, E)
