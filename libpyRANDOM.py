"""
pyRANDOM
library for random-number notebooks
2024-04-21
Georg Kaufmann
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import os,sys
import libpyRANDOM

#================================#
def tspSetupCoordinatesSquare(plot=False):
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
def tspCreateDistanceMatrix(x,y):
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
def dlaCreateGrid(nx,ny,minx=-1,maxx=1,miny=-1,maxy=1):
    """
    Diffusion-limited aggregation
    setup of model grid in [-1,1] domain
    input:
      nx,ny   - number of grid points in x,y direction
      minx,maxx - min/max of x side
      miny,maxy - min/max of y side
    output:
      X,Y       - 2D arrays of coordinates
      matrix    - 2D array of state of grid points
    """
    x = np.linspace(minx,maxx,nx)
    y = np.linspace(miny,maxy,ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    matrix = np.zeros_like(X)
    return X,Y,matrix


#================================#
def dlaCreateBC(nx,ny,matrix,location='center'):
    """
    Diffusion-limited aggregation
    setup of boundary conditions
    input:
      nx,ny     - number of grid points in x,y direction
      matrix    - 2D array of state of grid points
      location  - marker for seed location
    output:
      matrix    - 2D array of state of grid points (modified)
    """
    if (location == 'center'):
        # boundary nodes
        matrix[:,0]    = -2
        matrix[:,ny-1] = -2
        matrix[0,:]    = -2
        matrix[nx-1,:] = -2
        # seed nodes
        matrix[int(nx/2),int(ny/2)] = 1
    elif (location == 'bottom'):
        # boundary nodes
        matrix[:,ny-1] = -2
        # seed nodes
        matrix[:,0] = 1
    return matrix


#================================#
def dlaPlot(isaved,matrix,path='div',name='dla'):
    """
    Diffusion-limited aggregation
    plot matrix
    input:
      isaved - savings counter
      matrix - current state of matrix
    output:
      (to file)
    """
    filename = path+'/'+name+f"-{isaved:04}.png"
    print('Figure saved: '+filename)
    plt.title(f"{isaved:04}")
    cbar1=plt.imshow(matrix[:,:].T,origin='lower',extent=(-1,1,-1,1),cmap='hot')
    plt.colorbar(cbar1)
    plt.savefig(filename)
    plt.close()
    return


#================================#
def dlaRun(nx,ny,nsaved=1000,ncount=10000,seed='area',location='center',path='div/dla',name='dla',**kwargs):
    """
    Diffusion-limited aggregation
    run complete model
    input:
      nx,ny   - number of grid points in x,y direction
      nsaved  - save interval
      ncount  - max number of steps taken
    output:
      saved_matrix - 2D array of state of grid points for saved steps
    """
    # define weights for left,top,right,bottom direction
    weights = [25,25,25,25]
    for i in kwargs:
        if (i=='weights'): weights = kwargs[i]
    if (sum(weights)>100): sys.exit('weights > 100')
    # check for directory for plotting
    if not os.path.isdir(path):
        os.mkdir(path)
    # create grid and set boundary conditions
    X,Y,matrix = libpyRANDOM.dlaCreateGrid(nx,ny)
    matrix = libpyRANDOM.dlaCreateBC(nx,ny,matrix,location=location)
    # create random-number generator
    rng = np.random.default_rng(seed=12)
    # define 3D array for saving time steps, fill with initial time step
    saved_matrix = np.array([matrix])
    # initial step
    icount = 0
    isaved = 0
    stop   = False
    libpyRANDOM.dlaPlot(isaved,matrix,path=path,name=name)
    # loop over random walks
    while (not stop):
        icount += 1
        # seed with initial point
        if (seed == 'area'):
            i = rng.integers(1,nx)
            j = rng.integers(1,ny)
        elif (seed == 'cloud'):
            i = rng.integers(2,nx-2)
            j = rng.integers(ny-4,ny)
        elif (seed == 'lineEven'):
            i = rng.integers(1,nx/2)*2
            j = ny-2
        point = np.array([[X[i,j],Y[i,j]]])
        # loop over single random point
        onEdge    = False
        onCrystal = False
        while (not onEdge and not onCrystal):
            # check if point is on edge, then mark ...
            if (i == 0): onEdge = True
            if (i == nx-1): onEdge = True
            if (j == 0): onEdge = True
            if (j == ny-1): onEdge = True
            # ... and test, if point is not on edge
            if (not onEdge):
                # check if point is close to cluster, then crystallize
                if (matrix[i-1,j]>=1):
                    onCrystal = True
                    matrix[i,j]=icount
                elif (matrix[i+1,j]>=1):
                    onCrystal = True
                    matrix[i,j]=icount
                elif (matrix[i,j-1]>=1):
                    onCrystal = True
                    matrix[i,j]=icount
                elif (matrix[i,j+1]>=1):
                    onCrystal = True
                    matrix[i,j]=icount
                # if new point has crystallized, check if its along edge, then stop
                if (onCrystal):
                    if (matrix[i-1,j]==-2): stop = True
                    if (matrix[i+1,j]==-2): stop = True
                    if (matrix[i,j-1]==-2): stop = True
                    if (matrix[i,j+1]==-2): stop = True
                    if (stop==True): print('Crystal reached border')
            # proceed with random walk, check if edge is reached
            if (not onCrystal and not onEdge):
                decide = rng.uniform()*100
                if (decide < weights[0]):
                    i = i -1
                elif (decide < weights[0]+weights[1]):
                    j = j +1
                elif (decide < weights[0]+weights[1]+weights[2]):
                    i = i +1
                else:
                    j = j -1
                if (not onEdge):
                    point = np.append(point,[[X[i,j],Y[i,j]]],axis=0)
        # stop, if maximum counter is reached
        if (icount > ncount):
            stop = True
        # save to saved_matrix
        if((icount % nsaved)==0):
            isaved += 1
            libpyRANDOM.dlaPlot(isaved,matrix,path=path,name=name)
            saved_matrix = np.append(saved_matrix,[matrix],axis=0)
    return saved_matrix


#================================#
