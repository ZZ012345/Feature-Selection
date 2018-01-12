#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from ConstructPairwiseDistance import ConstructPairwiseDistance
from ConstructAffinityMatrix import ConstructAffinityMatrix
from construct_W import construct_W
import GetData

a = np.array([[1,2], [2,3], [4,5]])
print a
print np.size(a, axis=0)
