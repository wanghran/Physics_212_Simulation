
# coding: utf-8

# In[1]:

import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt



# In[2]:

#Parameters
grid_n = 30

T_p = 0.75  # init Tree probability
B_p = 0.015  # init Burning Tree probability
S_p = 0.20  # Fire Spread probability
L_p = 0.0005  # Lightning-Fire probability

#Final Constants
EMPTY = 0
#GROWTH_RATE = 0 # ?? what assumption => NO GROWTH
TREE = 10
NEW_BURNING = 15
BURNING = 20
BURNING_RATE = 1
BURNED = 26


# In[3]:

tf = 40
forest_list = np.empty(tf + 1, dtype=np.object)


# In[4]:

def init_forest(n, T_p, B_p):
    forest = (nprnd.rand(n, n) < T_p) * TREE
    forest_burn = (((forest == TREE) * nprnd.rand(n, n)) > (1-B_p)) * NEW_BURNING
    np.place(forest, forest_burn > TREE, forest_burn[forest_burn >TREE])
    return forest


# In[5]:

forest = init_forest(grid_n, T_p, B_p)
forest_list[0] = forest


# In[6]:

def check_if_on_fire(forest, i, j):
    N, S, E, W = i-1, i+1, j+1, j-1
    NW, NE, SW, SE = (N, W), (N, E), (S, W), (S, E)
    if N >= 0:  # As long as the top neighbor is within the grid
        if forest[N, j] >= BURNING and forest[N, j] != BURNED:
            return True
    if S < grid_n:  # As long as the bottom neighbor is within the grid
        if forest[S, j] >= NEW_BURNING and forest[S, j] != BURNED:
            return True
    if E < grid_n:  # As long as the right neighbor is within the grid
        if forest[i, E] >= NEW_BURNING and forest[i, E] != BURNED:
            return True
    if W >= 0:  # As long as the left neighbor is within the grid
        if forest[i, W] >= BURNING and forest[i, W] != BURNED:
            return True
    if N>=0 and E<grid_n:
        if forest[NE] >= BURNING and forest[NE] != BURNED:
            return True
    if N>=0 and W >= 0:
        if forest[NW] >= BURNING and forest[NW] != BURNED:
            return True
    if S< grid_n and E<grid_n:
        if forest[SE] >= NEW_BURNING and forest[SE] != BURNED:
            return True
    if S< grid_n and W >= 0:
        if forest[SW] >= NEW_BURNING and forest[SW] != BURNED:
            return True
    return False


# In[7]:

for t in range(tf):
    current_forest = np.copy(forest)
    for i in range(grid_n):
        for j in range(grid_n):
            current = current_forest[i, j]
            if current >= BURNED:
                continue
            elif current >= BURNING:
                current_forest[i, j] = current + BURNING_RATE
            elif current == NEW_BURNING:
                current_forest[i, j] = BURNING
            elif current == TREE:
                if (check_if_on_fire(current_forest, i, j) and nprnd.random() < S_p) or nprnd.random() < L_p:
                    current_forest[i, j] = NEW_BURNING
            else:
                continue  # if growth_rate is zero
                current_forest[i, j] = current + GROWTH_RATE  # otherwise   
    forest_list[t + 1] = current_forest
    forest = current_forest


# In[8]:

#### plotting 2 figures
fig, axes = plt.subplots(1,3)
time, time_step = 0, 20
for ax in axes.flat:
    im = ax.matshow(forest_list[time], vmin=0, vmax=BURNED+1)
    ax.set_xlabel("after " + str(time) + " hours", fontsize = 12)
    time = time+time_step
fig.suptitle("Forest Fire Simulation for the Total Duration of " + str(tf) + " Hours", ha='center', fontsize=14)
fig.subplots_adjust(top=1.00)
cax = fig.add_axes([0.2, 0.15, 0.6, 0.05])
cbar = fig.colorbar(im, cax, orientation='horizontal')
fig.show()
fig.savefig('Forest_Fire_Simulation_at_%s_and_%s_Hours.png'
             % (time-(2*time_step), time-time_step), dpi=100)

