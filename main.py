#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import datetime
from GetData import GetData
from skfeature.utility.construct_W import construct_W
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS
from EntropyBasedFeatureRanking import EntropyBasedFeatureRanking
from skfeature.function.similarity_based import SPEC


'''
dataSet用于标示实验数据集
dataSet = 0:
dataSet = 1:

methodType用于标示实验算法
methodType = 0:
methodType = 1:
'''

# initialization
dataSet = 0
data = GetData(dataSet)
methodType = 1
print "Data Preparation finished."

timeStart = datetime.datetime.now()

# feature selection
if methodType == 0:
    # Laplacian Score
    kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
    W = construct_W(data, **kwrags_W)
    result = lap_score.lap_score(data, W=W)
    print result
elif methodType == 1:
    # MCFS
    kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
    W = construct_W(data, **kwrags_W)
    print "Affinity Matrix Construction finished."
    result = MCFS.mcfs(data, 2, W=W, n_clusters=2)
    print result
elif methodType == 2:
    # Entropy based Feature Ranking
    result = EntropyBasedFeatureRanking(data)
    print result
elif methodType == 3:
    # SPEC
    kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
    W = construct_W(data, **kwrags_W)
    result = SPEC.spec(data, style=-1, W=W)
    print result

timeEnd = datetime.datetime.now()
print "Run Time: ", timeEnd - timeStart
