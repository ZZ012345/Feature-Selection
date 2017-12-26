#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy.io
from ReadFile import ReadFile
import construct_W
import datetime
from ConstructPairwiseDistance import ConstructPairwiseDistance


data, label = ReadFile('mnist.txt')
print data
print data.shape
print type(data)
'''
kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
W = construct_W.construct_W(data, **kwrags_W)
print W
'''
begin = datetime.datetime.now()
weightMat = ConstructPairwiseDistance(data)
end = datetime.datetime.now()
print weightMat.shape
print 'Time ', end-begin
