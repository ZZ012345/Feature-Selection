#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def GetEntropy(X):
    """
    Dash M, Liu H. Feature Selection for Clustering. 2000:110-121.
    该函数计算一组输入数据X的熵

    Parameters
    ----------
    X: {numpy array}, shape (n_samples, n_features)

    Returns
    -------
    entropy: {float}
    """

    # 数据标准化
    I = X.max(axis=0) - X.min(axis=0)
    featureNum = X.shape[1]
    for i in range(0, featureNum):
        if I[i] == 0:
            I[i] = 1
    X_ = X / I
    D = pairwise_distances(X_)
    dataNum = D.shape[0]
    averageD = D.sum() / (dataNum * (dataNum - 1))
    alpha = np.log(2) / averageD
    S = np.e ** (- alpha * D)
    # 将相似度矩阵的对角线置为0.5，避免如果为1，在计算熵矩阵H时会出现非数
    for i in range(0, dataNum):
        S[i, i] = 0.5
    H = - S * np.log2(S) - (1 - S) * np.log2(1 - S)
    for i in range(0, dataNum):
        H[i, i] = 0
    entropy = H.sum()
    return entropy


def EntropyBasedFeatureRanking(X):
    """
    基于熵的特征排序算法
    输出数组entropySet，entropySet[i]表示去除特征i之后数据集的熵，越小表示该特征的重要性越低

    Parameters
    ----------
    X: {numpy array}, shape (n_samples, n_features)

    Returns
    -------
    entropySet: {np.ndarray}
    """

    featureNum = X.shape[1]
    entropySet = []
    # 每次去除一个特征，计算此时数据集的熵
    for i in range(0, featureNum):
        X_ = np.delete(X, i, axis=1)
        entropySet.append(GetEntropy(X_))
    return np.array(entropySet)
