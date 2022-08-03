import numba
import numpy as np
from scipy.ndimage import convolve

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

@numba.njit("(f4[:,:])(f4[:,:], f8)", nogil=True, parallel=True)
def update_metropolis(grid, temp):
    ''' Performs a single time-step update on a grid using the Metropolis algorithm 
    
    Chooses L^2 random points and performs the MCMC step

    Parameters
    ----------
    grid: array
        The spin lattice of 0s and 1s
    temp: float
        The temperature of the system
    
    Returns
    _______
    array[:, :]
        The updated grid
    '''
    
    L = grid.shape[0]
    beta = 1. / temp
    
    xs = np.random.randint(0, L, size=(L, L))
    ys = np.random.randint(0, L, size=(L, L))
    ds = 2 * np.pi * np.random.rand(L, L).astype(np.float32)

    for i in range(L):
        for j in range(L):
            x, y = xs[i, j], ys[i, j]      
            s = grid[x, y]
            nbrs = grid[(x+1)%L, y], grid[(x-1)%L, y], grid[x, (y+1)%L], grid[x, (y-1)%L]
            
            cost = 0.
            for n in nbrs: cost += np.cos(s - n) - np.cos(s + ds[i, j] - n)

            if cost < 0 or np.random.random() < np.exp(-cost * beta):
                grid[x, y] = s + ds[i, j]
                if grid[x, y] > 2 * np.pi: grid[x, y] -= 2 * np.pi  # clamp angles to [0, 2π) for visualisation
                elif grid[x, y] < 0: grid[x, y] += 2 * np.pi

    return grid

''' Defines the vorticity and magnetization operators on the lattice '''

kernel = [[True, True], [True, True]]
def vorticity(grid):
    return convolve(np.cos(grid), kernel, mode='wrap') ** 2 + \
           convolve(np.sin(grid), kernel, mode='wrap') ** 2

def magnetization(grid):
    return np.sum(np.cos(grid)) ** 2 + np.sum(np.sin(grid)) ** 2
