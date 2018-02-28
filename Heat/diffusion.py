import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

# PDE integrator
def diffusion(D, dt, dx, u):
#a: diffusion constant
#dt: step size in time
#dx: step size in space
#u: grid of diffusing quantity
#return: grid u after one time step 
    tmp = u[1:-1,1:-1]
    tmpu = u[0:-2,1:-1]
    tmpd = u[2:,1:-1]
    tmpl = u[1:-1,0:-2]
    tmpr = u[1:-1,2:]
    tmp = tmp + dt*D*(tmpu+tmpd+tmpl+tmpr-4*tmp)
    return tmp
def bound_cons(grid,T,dt):
    for i in range(0,T,dt):    
        grid= np.insert(grid,(0,grid.shape[0]),0,axis =0)
        grid= np.insert(grid,(0,grid.shape[1]),0,axis =1) 
        grid = diffusion(D,dt,dx,grid)
        if ((i+1)%20 == 0):
             plt.imshow(grid)
             plt.colorbar(orientation='vertical')
             plt.show()
    
    return grid
def bound_ref(grid,T,dt):
    for i in range(0,T,dt):
        grid = np.insert(grid,(0,grid.shape[0]),grid[(0,-1),:],axis = 0)
        grid = np.insert(grid,(0,grid.shape[1]),grid[:,(0,-1)],axis = 1)
        grid = diffusion(D,dt,dx,grid)
        if ((i+1)%20 == 0):
             plt.imshow(grid)
             plt.colorbar(orientation='vertical')
             plt.show()

    return grid
def bound_per(grid,T,dt):
    for i in range(0,T,dt):
        grid = np.insert(grid,(0,grid.shape[0]),grid[(-1,0),:],axis =0)
        grid = np.insert(grid,(0,grid.shape[1]),grid[:,(-1,0)],axis = 1)
        grid = diffusion(D,dt,dx,grid)
        if ((i+1)%20 == 0):
             plt.imshow(grid)
             plt.colorbar(orientation='vertical')
             plt.show()

    return grid
# set which code parts to use to compute solution
# define integration step sizes
dx = 1
dt = 1

L = 100   # system shape is square for convenience
T = 100   # integration time
D = 0.1   # diffusion constant

# initial heat distribution
grid = np.zeros(2 * [L * int(dx**-1)])
gridsize = grid.shape
grid[int(gridsize[0] / 2), int(gridsize[1] / 2)] = 75
grid[int(gridsize[0] / 4), int(gridsize[1] / 4)] = 35
#### sequential processing solution
def sequential():
    grid_s = np.copy(grid)  # keep original grid variable unchanged
    ts = time.time()# measure computation time
    print("absorbing boundary:")
    grid_1 = bound_cons(grid_s,T,dt)
    print("reflecting boundary:")
    grid_2 = bound_ref(grid_s,T,dt)
    print("periodic boundary:")
    grid_3 = bound_per(grid_s,T,dt)
    print('Sequential processing took {}s'.format(time.time() - ts))

    ## PUT HERE YOUR SEQUENTIAL CODE SOLUTION ##

    return (grid_1,grid_2,grid_3)
    
#### concurrent processing solution
def parallel(n,grid,Type):
    # define number of processes
    units = n
    p = Pool(units)

    # define how many partitions of grid in x and y direction and their length
    (nx, ny) = (int(units / 2), 2)
    lx = int(gridsize[0] / nx)
    ly = int(gridsize[1] / ny)

    # this makes sure that D, dt, dx are the same when distributed over processes
    # for integration, so the only interface parameter that changes is the grid
    func = partial(diffusion, D, dt, dx)
    
    for t in np.arange(T/dt):  # note numpy.arange is rounding up floating points
        data = []
        # prepare data to be distributed among workers
        # 1. insert boundary conditions and partition data
        
        if (Type ==1):
            grid = np.insert(grid, (0, gridsize[0]), 0, axis=0)
            grid = np.insert(grid, (0, gridsize[1]), 0, axis=1)
        if(Type ==2):    
            grid = np.insert(grid, (0, gridsize[0]), grid[(0, -1), :], axis=0)
            grid = np.insert(grid, (0, gridsize[1]), grid[:, (0, -1)], axis=1)
        if (Type ==3):
            grid = np.insert(grid, (0, gridsize[0]), grid[(-1, 0), :], axis=0)
            grid = np.insert(grid, (0, gridsize[1]), grid[:, (-1, 0)], axis=1) 
        # partition into subgrids
        for i in range(nx):
            for j in range(ny):
                # subgrid
                subg = grid[i * lx + 1:(i+1) * lx + 1, j * ly + 1:(j+1) * ly + 1]
                # upper boundary
                subg = np.insert(subg, 0, grid[i * lx, j * ly + 1:(j+1) * ly + 1],
                                 axis=0)
                # lower boundary
                subg = np.insert(subg, subg.shape[0],
                                 grid[(i+1) * lx + 1, j * ly + 1:(j+1) * ly + 1],
                                 axis=0)
                # left boundary
                subg = np.insert(subg, 0, grid[i * lx:(i+1) * lx + 2, j * ly],
                                 axis=1)
                # right boundary
                subg = np.insert(subg, subg.shape[1],
                                 grid[i * lx:(i+1) * lx + 2, (j+1) * ly + 1],
                                 axis=1)
                # collect subgrids in list to be distributed over processes
                data.append(subg)
        # 2. divide among workers
        
        results = p.map(func, data)
        grid = np.vstack([np.hstack((results[i * ny:(i+1) * ny])) for i in range(nx)])
        if ((t+1)%20 == 0):
             plt.imshow(grid)
             plt.colorbar(orientation='vertical')
             plt.show()

    
    # plot grid
    '''
    plt.imshow(grid)
    plt.show()
    '''
    return grid
    
sequential()
ts = time.time()  # measure computation time
n = 8 # number of process
print("parallel absorbing boundary:")
grid_1s = parallel(n,grid,1) #absorbing boundary
print("parallel reflecting boundary:")
grid_2s = parallel(n,grid,2) #reflecting boundary
print("parallel periodic boundary:")
grid_3s = parallel(n,grid,3) # periodic boundary
print('Parallel processing took {}s'.format(time.time() - ts))


