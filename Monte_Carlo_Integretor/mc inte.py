# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 02:11:54 2016

@author: Rex
"""

import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt

def to_integr(x):
    return x**2

# monte carlo integrator
def mc_integr(func, a, b, fmax, N):
    x = a+(b-a)*nprnd.random(N)
    y = fmax*nprnd.random(N)
    counts_under_curve = np.sum(y <= func(x))
    frac_under_curve = counts_under_curve/N
    area = (b-a)*fmax*frac_under_curve
    return area


# area using monte carlo integrator
area = mc_integr(to_integr, 0, 1, 1, 100000)
n_trials = 10
N = np.logspace(1, 5, 9)
areaN = np.zeros((N.size, n_trials))

for i in np.arange(N.size):
    for j in np.arange(n_trials):
        areaN[i, j] = mc_integr(to_integr, 0, 1, 2, N[i])


# plotting areas as a function of N
plt.semilogx(N, areaN, 'o')
plt.xlabel('N')
plt.ylabel('area')
plt.show()

# plotting the standard deviation of the area as a function of N
# and comparing to 1/sqrt(N)
plt.loglog(N, np.std(areaN, 1), 'o')
plt.loglog(N, 1/np.sqrt(N))
plt.xlabel('N')
plt.ylabel('standard deviation of the area')
plt.show()
std = np.std(areaN, 1)
