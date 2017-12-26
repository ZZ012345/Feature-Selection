#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
from ConstructPairwiseDistance import ConstructPairwiseDistance
from sklearn.metrics.pairwise import pairwise_distances
import datetime
import construct_W
from skfeature.function.sparse_learning_based.MCFS import mcfs


mat = scipy.io.loadmat("COIL20.mat")
X = mat['X']
kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
W = construct_W.construct_W(X, **kwrags_W)
print W
weightMat = mcfs(X, 10, **{"W": W, "n_clusters": 20})
print weightMat
print weightMat.shape
np.savetxt("a.txt", weightMat, fmt='%.5f')
