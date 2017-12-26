#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


def ReadFile(fileName):
    data = []
    label = []
    # 打开文件
    with open('./' + fileName, 'r') as f:
        # 按行读取
        for line in f.readlines():
            # 按逗号分隔提取数据
            lineData = line.strip().split(',')
            count = 0
            lineLen = len(lineData)
            fLineData = []
            while count < lineLen - 1:
                fLineData.append(float(lineData[count]))
                count = count + 1
            # 存储标签
            label.append(int(lineData[count]))
            # 存储特征
            data.append(fLineData)

    return np.array(data), label