#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from sklearn.datasets import make_swiss_roll


def GetData(dataSet):
    """
    该函数根据输入类型dataType生成或读取相应数据

    Parameters
    ----------
    dataType: 数据类型

    Returns
    -------
    dataSet: 数据集
    """

    newData = False
    if dataSet == 0:
        # 3维人工数据集,前2维满足高斯分布,最后1维随机分布
        if newData:
            sampleNum = 20000
            mean1 = [0, 0]
            cov1 = [[10, 5], [5, 5]]
            x1, y1 = np.random.multivariate_normal(mean1, cov1, sampleNum/2).T
            mean2 = [15, 0]
            cov2 = [[10, 5], [5, 5]]
            x2, y2 = np.random.multivariate_normal(mean2, cov2, sampleNum/2).T
            z = [random.uniform(-10, 10) for _ in range(sampleNum)]
            data = np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2)), z)).T
            np.savetxt("./data/Gaussian.txt", data, fmt='%.5f')
        else:
            data = np.loadtxt("./data/Gaussian.txt")

        return data

    elif dataSet == 1:
        # swiss roll数据集
        if newData:
            sampleNum = 20000
            data, color = make_swiss_roll(sampleNum)
            dataToSave = np.hstack((data, color.reshape(sampleNum, 1)))
            np.savetxt("./data/SwissRoll.txt", dataToSave, fmt='%.5f')
        else:
            dataSaved = np.loadtxt("./data/SwissRoll.txt")
            data = dataSaved[:, 0:3]

        return data
