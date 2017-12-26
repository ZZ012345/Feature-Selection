#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from EntropyBasedFeatureRanking import EntropyBasedFeatureRanking


a = np.array([[1, 2, 3], [4, 2, 6], [2, 3, 4], [0, 0, 0]], dtype=float)
result = EntropyBasedFeatureRanking(a)
print result
