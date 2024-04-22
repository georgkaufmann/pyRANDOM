"""
pyRANDOM
library for random-number notebooks
2024-04-21
Georg Kaufmann
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools

#================================#
def setupCoordinatesSquare(plot=False):
    """
    define 2D coordinates in a square 
    four corner as points, plus central point, slightly offset
    input:
      (none)
    output:
      - x,y [m] - coordinates within range [-1,1]
      - cities  - name of node (counter)
    """
    N      = 5
    x      = np.array([-1,-1,1,1,0.0])
    y      = np.array([-1,1,1,-1,0.2])
    cities = np.arange(N)
    
    if (plot):
        plt.figure(figsize=(6,6))
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.plot(x,y,lw=0,marker='o',markersize=15)
        for i in range(len(x)):
            plt.text(x[i],y[i],str(cities[i]),ha='center',va='center',color=(1,1,1))
    return x,y,cities


#================================#
def createDistanceMatrix(x,y):
    """
    create distance matrix between points
    pytagorean distance between points i and i+1
    for entries i,i (node connected to itself), distance is zero
    input:
      x,y [m] - coordinates of nodes
      N       - number of points
    output:
      matrix  - distance matrix
    """
    N = x.shape[0]
    matrix = np.zeros(N*N).reshape(N,N)
    for i in range(N):
        for j in range(N):
            matrix[i][j] = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
    return matrix


#================================#
def tspPermutation(cities,matrix):
    """
    Solve traveling-salesmen problem
    version with permutations
    input:
      cities - list of names for points
      matrix - matrix of distances between two points
    output:
      total_dist_min [m] - minimum distance
      total_dist_max [m] - maximum distance
      minorder           - list of points of shortest path
    """
    # get number of cities
    N = cities.shape[0]
    print(cities)
    # create all permutations for distances
    perm = itertools.permutations(cities[1:], N-1)
    # start loop over permutations
    icount         = 0
    total_dist_min = np.inf
    total_dist_max = 0.
    # loop over permutations
    for order in list(perm):
        icount += 1
        # add start/end as zero index
        neworder = (0,)+order+(0,)
        #print (neworder,type(neworder),len(neworder))
        total_dist = 0.
        # calculate total_dist, exclude "infinite" connections
        for i in range(len(neworder)-1):
            if (np.isnan(matrix[neworder[i]][neworder[i+1]])):
                dist = np.inf
            else:
                dist = matrix[neworder[i]][neworder[i+1]]
            total_dist += dist
        # check if total_dist is smallest value
        #minorder = 0.
        if (total_dist < total_dist_min):
            total_dist_min = total_dist
            minorder = neworder
        if (total_dist > total_dist_max and total_dist != np.inf):
            total_dist_max = total_dist
    # Best result
    print('min path: ',total_dist_min)
    print('max path: ',total_dist_max)
    print('best path: ',minorder,icount)
    return total_dist_min,total_dist_max,minorder


#================================#
def tspShuffle(cities,matrix):
    """
    Solve traveling-salesmen problem
    version with random shuffling
    input:
      cities - list of names for points
      matrix - matrix of distances between two points
    output:
      total_dist_min [m] - minimum distance
      total_dist_max [m] - maximum distance
      minorder           - list of points of shortest path
    """
    # get number of cities
    N = cities.shape[0]
    print(cities)
    # define max number of shuffles
    M = np.math.factorial(N-1)

    total_dist_min = np.inf
    total_dist_max = 0.
    # loop over permutations
    for icount in range(1,M):
        np.random.shuffle(cities[1:])
        neworder = np.append(cities,0)
        #print (neworder,type(neworder),len(neworder))
        total_dist = 0.
        # calculate total_dist, exclude "infinite" connections
        for i in range(len(neworder)-1):
            if (np.isnan(matrix[neworder[i]][neworder[i+1]])):
                dist = np.inf
            else:
                dist = matrix[neworder[i]][neworder[i+1]]
            total_dist += dist
        # check if total_dist is smallest value
        if (total_dist < total_dist_min):
            total_dist_min = total_dist
            minorder = neworder
        if (total_dist > total_dist_max and total_dist != np.inf):
            total_dist_max = total_dist
    # Best result
    print('min path: ',total_dist_min)
    print('max path: ',total_dist_max)
    print('best path: ',minorder,icount)
    return total_dist_min,total_dist_max,minorder


#================================#
def tspNaturalNeigbours(matrix):
    """
    Solve traveling-salesmen problem
    version with natural neighbors
    input:
      matrix - matrix of distances between two points
    output:
      total_dist_min [m] - minimum distance
      total_dist_max [m] - maximum distance
      minorder           - list of points of shortest path
    """
    # get number of cities
    N = matrix.shape[0]
    # start from first location, put on stack best_path
    istart    = 0 #1
    i         = istart
    best_path = [i]
    # loop over remaining cities
    for n in range(1,N):
        rest = np.argsort(matrix[i][:])
        #print(rest)
        for j in range(1,len(rest)):
            onstack = False
            for k in range(len(best_path)):
                #print(n,j,k,best_path[k],rest[j])
                if (best_path[k]==rest[j]):
                    onstack = True
            if (onstack==False):
                best_path.append(rest[j])
                i = rest[j]
                break
    # print best result found
    print(best_path)
    best_path = np.r_[best_path,istart]
    # calculate distance
    total_dist_min = 0
    for i in range(1,len(best_path)):
        total_dist_min += matrix[best_path[i-1]][best_path[i]]
    print('min path: ',total_dist_min)
    print('best path: ',best_path)
    return total_dist_min,best_path


#================================#
#================================#
