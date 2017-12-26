#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from ConstructPairwiseDistance import ConstructPairwiseDistance
from ConstructAffinityMatrix import ConstructAffinityMatrix
from construct_W import construct_W


a = np.array([[1, 0], [2, 0], [4, 0], [3, 0], [2, 2], [1, 2], [5, 0], [0, 5]])
b = ConstructAffinityMatrix(a)
print b
e = np.argsort(b, axis=1)
print e
kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 2, "t": 1}
d = construct_W(a, **kwrags_W)
print d

