#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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

    figure = False
    if dataSet == 0:
        # 3维人工数据集,前2维满足正态分布,最后1维随机分布
        sampleNum = 5000
        mean1 = [0, 0]
        cov1 = [[10, 5], [5, 5]]
        x1, y1 = np.random.multivariate_normal(mean1, cov1, sampleNum).T
        mean2 = [15, 0]
        cov2 = [[10, 5], [5, 5]]
        x2, y2 = np.random.multivariate_normal(mean2, cov2, sampleNum).T
        z = [random.uniform(-10, 10) for _ in range(sampleNum * 2)]

        if (figure):
            fig = plt.figure()
            plot1 = fig.add_subplot(121, projection='3d')
            plot1.scatter(x1, y1, z[0:sampleNum], c='r', marker='o')
            plot1.scatter(x2, y2, z[sampleNum:sampleNum*2], c='b', marker='x')
            plot1.set_xlabel('x axis')
            plot1.set_ylabel('y axis')
            plot1.set_zlabel('z axis')
            plot2 = fig.add_subplot(122)
            plot2.scatter(x1, y1, c='r', marker='o')
            plot2.scatter(x2, y2, c='b', marker='x')
            plot2.set_xlabel('x axis')
            plot2.set_ylabel('y axis')
            plt.show()

        return np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2)), z)).T
        