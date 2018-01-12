#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import datetime
from skfeature.utility.construct_W import construct_W
from skfeature.function.sparse_learning_based import MCFS
import ConstructWbyEdge

dataSet = 1
useEdge = False

if dataSet == 0:
    data = np.loadtxt("./data/GaussianTopologyNode.txt")
    edge = np.loadtxt("./data/GaussianTopologyEdge.txt")
    timeStart = datetime.datetime.now()

    if useEdge:
        W = ConstructWbyEdge.ConstructWbyEdge(data, edge, t=1)
    else:
        kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 3, "t": 1}
        W = construct_W(data, **kwrags_W)
    result = MCFS.mcfs(data, n_selected_features=2, W=W, n_clusters=2)
    print result

    timeEnd = datetime.datetime.now()
    print "Run Time: ", timeEnd - timeStart

elif dataSet == 1:
    data = np.loadtxt("./data/SwissRollTopologyNode.txt")
    edge = np.loadtxt("./data/SwissRollTopologyEdge.txt")
    timeStart = datetime.datetime.now()

    if useEdge:
        W = ConstructWbyEdge.ConstructWbyEdge(data, edge, t=1)
    else:
        kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 3, "t": 1}
        W = construct_W(data, **kwrags_W)
    result = MCFS.mcfs(data, n_selected_features=2, W=W, n_clusters=2)
    print result

    timeEnd = datetime.datetime.now()
    print "Run Time: ", timeEnd - timeStart
