import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
global y1
def od1():
     t_min = 0; t_max1 = 60; t_max2 = 500;dt = 1
     t1 = np.arange(t_min, t_max1+dt, dt)
     t2 = np.arange(t_max1,t_max2+dt,dt)
     initial_conditions = [(5000.0,0.0, 0.0)]
     I=200.0
     g=9.8
     dmdt=4000/60
     parameters=(I,g,dmdt)
     plt.figure() # Create figure; add plots later.
     y1=run1(initial_conditions,parameters,t1)
     y2=y1[-1]
     a = y2[0]
     b = y2[1]
     c = y2[2]
     t = np.concatenate((t1,t2),axis=0)
     new_conditions=[(a,b,c)]
     y2 = run2(new_conditions,t2)
     Y = np.concatenate((y1,y2),axis=0)
     Y2 =  [row[2] for row in Y]
     Y3=Y2*(1+.05*np.random.randn(len(Y2)))
     np.save('t_value',t)
     np.save('x_value',Y3)
     return Y2[500]
def run1(initial_conditions, para,t1):
    for y0 in initial_conditions:
        y = odeint(F, y0, t1, args=para)
        plt.subplot(211)
        plt.plot(t1, y[:, 1], linewidth=2)
        plt.subplot(212)
        plt.plot(t1,y[:,2],linewidth=2)
    return y
def run2(new_conditions, t2):
    for y1 in new_conditions:
        y2 = odeint(F2,y1,t2)
        plt.subplot(211)
        plt.plot(t2,y2[:,1],linewidth = 2)
        plt.subplot(212)
        plt.plot(t2,y2[:,2],linewidth = 2)
    return y2
def F(y,t,I,g,dmdt):
    m = y[0]
    v =y[1]
    return -dmdt,I*g*dmdt/m, v

def F2(y,t):
    v = y[1]
    return 0,-9.8,v

