# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def Whencanibuyit(p1,p2,s,pay,r1,r2,r3):      
    SimulationTime = 50 
    dt = 1
    Time = np.arange(dt,SimulationTime)
    NewCar = np.zeros(Time.size)
    OldCar = np.zeros(Time.size)
    Saving = np.zeros(Time.size)
    Money = np.zeros(Time.size)
    NewCar[0] = p1
    OldCar [0] = p2
    Saving [0] = s
    Money[0] = OldCar[0] + Saving[0]
    Payment = pay
    Rate1 = r1
    Rate2 = r2
    Rate3 = r3
    
    for i in np.arange(1, SimulationTime):
        NewCar[i] = NewCar[i-1] + dt * Rate1 * NewCar[i-1]
        OldCar[i] = OldCar[i-1] - dt * Rate2 * OldCar[i-1]
        Saving[i] = Saving[i-1] * (1 + Rate3) + Payment
        Money[i] = OldCar[i] + Saving[i]
        if (Money[i] > NewCar[i] and Money[i-1] < NewCar[i-1]):
            print(i)
          
        
    plt.plot(Time, NewCar, label = 'price of new car')
    plt.plot(Time, Money, label = 'Total Saving')
    plt.legend()
    plt.xlabel = ('Time')
    plt.ylabel = ('Price')
    plt.show()
