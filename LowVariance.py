#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import fisher_score
from sklearn import svm
from sklearn.metrics import accuracy_score
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from skfeature.utility import unsupervised_evaluation

mat = scipy.io.loadmat("COIL20.mat")
print type(mat)
X = mat['X']
print X
print type(X)
n_samples, n_features = np.shape(X)
print n_samples, n_features
label = mat['Y']
y = label[:, 0]
n_labels = np.shape(y)
print n_labels
kwrags_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, "t": 1}
W = construct_W.construct_W(X, **kwrags_W)
print W
print type(W)
score = lap_score.lap_score(X, W=W)
print score
idx = lap_score.feature_ranking(score)
print idx
num_feature = 5
selected_features = X[:, idx[0:num_feature]]
print selected_features
num_cluster = len(np.unique(y))
print num_cluster
nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
print nmi
print acc
