#!/usr/bin/env python
import numpy as np

m_list = list()
m_list[0].append([1, 2, 3])
m_list[0].append([1, 2, 3])




print m_list
cov = np.cov(m_list)
print cov