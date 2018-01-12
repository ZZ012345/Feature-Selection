#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import datetime
from GetData import GetData
from skfeature.utility.construct_W import construct_W
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS
from EntropyBasedFeatureRanking import EntropyBasedFeatureRanking
from skfeature.function.similarity_based import SPEC


# initialization
methodType = 0
dataSet = 0
data = GetData(dataSet)
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
    # 参数n_selected_features用于控制LARs算法解的稀疏性，也就是result每一列中非零元素的个数
    # 参数n_clusters用于控制LE降维的目标维数，也就是result的列数
    result = MCFS.mcfs(data, n_selected_features=2, W=W, n_clusters=2)
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
print "Run Time:", timeEnd - timeStart
