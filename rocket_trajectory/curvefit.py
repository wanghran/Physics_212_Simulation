# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:15:54 2016

@author: PeteWu
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fit():
    t = np.load('t_value.npy')
    x = np.load('x_value.npy')
    A = np.vstack([np.ones(len(t)), t, t**2, t**3, t**4, t**5]).T
    a, b, c, d, e,f = np.linalg.lstsq(A, x)[0]
    plt.plot(t, x, 'o', label='Original data', markersize=2)
    t = np.arange(-0,500,1)
    val2 = a+b*t+c*t**2+d*t**3+e*t**4+f*t**5
    plt.plot(t, val2, 'r', label='Fitted line')
    plt.legend(loc=4)
    plt.show()
    return val2[-1]


init_conc = 0.0

t = np.load('t_value.npy')
t1 = t[0:60]
t2 = t[60:]
x = np.load('x_value.npy')   
x1 = x[0:60]
x2 = x[60:] 
init_conc2=x[59]
def F(x,t,initil_mass,roc_mass,burn_t,i):
    dmdt = (initil_mass-roc_mass)/burn_t
    m = initil_mass - dmdt*t
    return (-9.8+i*9.8*dmdt)*t/m
    
def F2(t, initil_mass,roc_mass,burn_t,i):
    return odeint(F, init_conc, t, args=(initil_mass,roc_mass,burn_t,i)).flatten()

#def logF2(t,initil_mass,roc_mass,burn_t,i):
   # return np.log(odeint(F, init_conc, t, args=(initil_mass,roc_mass,burn_t,i)).flatten())

def F3(x,t,v,g):
    return v-g*t
    
def F4(t,v,g):
    return odeint(F3,init_conc2,t,args=(v,g)).flatten()

a,b,c,d = curve_fit(F2, t1, x1,p0=(5381,1221,60,180))[0]

v,g = curve_fit(F4,t2,x2,p0=(4000,9))[0]
plt.plot(t, x,'o', label='data',markersize=2)

plt.plot(t1, F2(t1, a,b,c,d), label='1st fit',linewidth=2,color='r')
plt.plot(t2, F4(t2, v,g), label='2rd fit',linewidth=2,color='y')
plt.legend()
plt.show()
val1 = F4(t2,v,g)
print(str(val1[-1]))
val2=fit()
print(str(val2))
print('Fitted a=' + str(a))
print('Fitted b=' + str(b))
print('fitted c=' + str(c))
print('fitted d=' + str(d))
print('fitted v=' + str(v))
print('fitted g=' + str(g))
