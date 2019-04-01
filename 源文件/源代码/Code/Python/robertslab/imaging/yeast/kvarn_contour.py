"""
Performs contour extraction of yeast cells using the algorithm in:

Image analysis algorithms for cell contour recognition in budding yeast
Mats Kvarnstrom, Katarina Logg, Alfredo Diez, Kristofer Bodvard, Mikael Kall
"""

from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage import io,filter,draw,img_as_float
import robertslab.imaging.imaging as imaging
import robertslab.imaging.yeast.contour as contour
import robertslab.imaging.yeast.segment as segment

def findCellContour(I, Ix, Iy, xc, yc, rmax, N=200, M=100, showIntermediates=False):
    
    # Fill in the cost matrix.
    (rs,thetas,C)=constructCostMatrix(I, xc, yc, rmax, N, M)
    
    # Find the optimal path.
    path=findOptimalPath(C)
    
    # Figure out the rs from the path.
    rpath=np.zeros((len(path),),dtype=float)
    for i in range(0,len(path)):
        rpath[i]=rs[path[i]]
        
    # Create the contour object.
    c = contour.CellContour(xc, yc, thetas, rpath, I, Ix, Iy)
    
    if showIntermediates:
        plt.figure()
        plt.subplot(1,2,1)  
        io.imshow(I)
        plt.scatter(c.xc,c.yc,marker='x',color='g')
        plt.plot(c.rs*c.coss+c.xc,c.rs*c.sins+c.yc,marker='x',color='r')
        plt.axis([0,I.shape[1],I.shape[0],0])
        plt.subplot(1,2,2)  
        io.imshow(C)
        plt.plot(range(0,len(path)),path,marker='x',color='g')
        plt.axis([0,C.shape[1],0,C.shape[0]])
        io.show()
        
    return c
        
def constructCostMatrix(im, xc, yc, rMax, N, M):
    
    # Calculate the image gradient.
    (Iy,Ix)=segment.calculateHVGradients(im)
    
    # Figure out the position of the matrix elements. 
    rs=np.arange(0.0, rMax, rMax/float(N))
    thetas=np.arange(0.0, 2*np.pi, 2*np.pi/float(M))
    coss=np.cos(thetas)
    sins=np.sin(thetas)
    C=np.zeros((N,M),dtype=np.float64)
    
    # Fill in the costs using the radial derivative of the image gradient.
    for i in range(1,len(rs)-1):
        for j in range(0,len(thetas)):
            xm=rs[i-1]*coss[j]+xc
            xp=rs[i+1]*coss[j]+xc
            ym=rs[i-1]*sins[j]+yc
            yp=rs[i+1]*sins[j]+yc
            C[i,j]=-((imaging.bilinearInterpolate(Ix,xp,yp)-imaging.bilinearInterpolate(Ix,xm,ym))*coss[j] + (imaging.bilinearInterpolate(Iy,xp,yp)-imaging.bilinearInterpolate(Iy,xm,ym))*sins[j])
        
    return (rs,thetas,C)

def findOptimalPath(CO, pb=10):
    
    # Create a cost matrix using a certain number of the columns as periodic boundaries.
    C=np.zeros((CO.shape[0],CO.shape[1]+pb*2),dtype=np.float64)
    C[:,0:pb] = CO[:,CO.shape[1]-pb:]
    C[:,pb:C.shape[1]-pb] = CO
    C[:,C.shape[1]-pb:] = CO[:,0:pb]
    N=C.shape[0]
    M=C.shape[1]
    Q=np.zeros(C.shape,dtype=np.float64)
    P=np.zeros(C.shape,dtype=np.int32)
    
    # Initialize the first column of Q.
    Q[:,0]=C[:,0]
    
    # Loop over each column.
    for m in range(1,M):
        
        # Loop over each row.
        for i in range(0,N):
            
            # Find the minimum cost from transitioning here from each row of the previous column.
            minCost = float("inf")
            minCostIndex = -1
            for k in range(0,N):
                cost = Q[k,m-1] + getTransitionCost(k,i)
                if cost < minCost:
                    minCost = cost
                    minCostIndex = k
            
            # Save the total minimum cost to get here into Q and also the transition made into P.
            Q[i,m] = C[i,m]+minCost
            P[i,m] = minCostIndex
           
    # Reconstruct the optimal path in reverse.
    path=[]
    minCost = float("inf")
    minCostIndex = -1
    for k in range(0,N):
        if Q[k,M-1] < minCost:
            minCost = Q[k,M-1]
            minCostIndex = k
    path.append(minCostIndex)
    for m in range(M-1,0,-1):
        path.append(P[path[-1],m])
        
    # Flip the path so it is in the forward direction.
    path.reverse()
            
    # Return the portion of the path excluding the periodic boundaries.
    return path[pb:len(path)-pb]
    
def getTransitionCost(k,i):
    return 0.001*((k-i)**2)


