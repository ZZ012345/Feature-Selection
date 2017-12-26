#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from ConstructPairwiseDistance import ConstructPairwiseDistance
from ConstructAffinityMatrix import ConstructAffinityMatrix
from construct_W import construct_W

a = np.array([[0, 1, 0], [0, 0, 0], [2, 0, 0]])
print a
b = np.transpose(a) > a
print b
c = a.multiply(b)
