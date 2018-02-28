# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:22:16 2016

@author: Rex
"""

import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt
r=nprnd.random(1000)
phi=2*np.pi*nprnd.random(1000)
x=np.sqrt(r)*np.cos(phi)
y=np.sqrt(r)*np.sin(phi)
plt.plot(x,y,'o')


newx = -1 + 2 * nprnd.random(1000)
newy = -1 + 2 * nprnd.random(1000)

for i in range(1000):
    if (newx[i] ** 2 + newy[i] ** 2 > 1):
        np.delete(newx,i)
        np.delete(newy,i)
plt.plot(newx,newy,'x')