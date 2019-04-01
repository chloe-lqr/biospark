"""
Performs contour extraction of yeast cells using the algorithm in:

"""

from __future__ import division
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.ndimage as ndimage
import scipy.optimize as opt
from skimage import io,draw,img_as_float
import robertslab.imaging.imaging as imaging
import robertslab.imaging.yeast.segment as segment

class CellContour:

    @staticmethod
    def doCellsOverlap(c1, c2, border=0.0):

        # If the bounding boxes do not overlap anywhere, the cell must not overlap.
        if c1.bounds[0]+c1.Ix0-border > c2.bounds[2]+c2.Ix0+border or c1.bounds[2]+c1.Ix0+border < c2.bounds[0]+c2.Ix0-border or c1.bounds[1]+c1.Iy0-border > c2.bounds[3]+c2.Iy0+border or c1.bounds[3]+c1.Iy0+border < c2.bounds[1]+c2.Iy0-border:
            return False

        # Go through each point in cell 1 and see if it falls in cell 2.
        for i in range(0,len(c1.rs)):
            if c2.containsPoint(c1.xs[i]+c1.Ix0, c1.ys[i]+c1.Iy0, border=border):
                return True

        # Go through each point in cell 2 and see if it falls in cell 1.
        for i in range(0,len(c2.rs)):
            if c1.containsPoint(c2.xs[i]+c2.Ix0, c2.ys[i]+c2.Iy0, border=border):
                return True
        return False

    @staticmethod
    def getOverlapingRadii(c1, c2, border=0.0):

        # If the bounding boxes do not overlap anywhere, the cell must not overlap.
        if c1.bounds[0]+c1.Ix0-border > c2.bounds[2]+c2.Ix0+border or c1.bounds[2]+c1.Ix0+border < c2.bounds[0]+c2.Ix0-border or c1.bounds[1]+c1.Iy0-border > c2.bounds[3]+c2.Iy0+border or c1.bounds[3]+c1.Iy0+border < c2.bounds[1]+c2.Iy0-border:
            return False

        overlappingRs=[]

        # Go through each point in cell 1 and see if it falls in cell 2.
        for i in range(0,len(c1.rs)):
            if c2.containsPoint(c1.xs[i]+c1.Ix0, c1.ys[i]+c1.Iy0, border=border):
                overlappingRs.append((0,i))

        # Go through each point in cell 2 and see if it falls in cell 1.
        for i in range(0,len(c2.rs)):
            if c1.containsPoint(c2.xs[i]+c2.Ix0, c2.ys[i]+c2.Iy0, border=border, showIntermediates=True):
                overlappingRs.append((1,i))

        return overlappingRs


    def __init__(self, xc, yc, thetas, rs, I, Ix0, Iy0):
        self.id = None
        self.xc = xc
        self.yc = yc
        self.thetas = thetas
        self.rs = rs
        self.coss = np.cos(thetas)
        self.sins = np.sin(thetas)
        self.xs = self.rs*self.coss+self.xc
        self.ys = self.rs*self.sins+self.yc
        self.bounds = (np.min(self.xs),np.min(self.ys),np.max(self.xs),np.max(self.ys))
        self.I = I
        self.Ix0 = Ix0
        self.Iy0 = Iy0

    def fillImageMask(self, Imask, border=0.0):
        for x in range(int(math.floor(self.bounds[0]-border))+self.Ix0,int(math.ceil(self.bounds[2]+border))+self.Ix0+1):
            for y in range(int(math.floor(self.bounds[1]-border))+self.Iy0,int(math.ceil(self.bounds[3]+border))+self.Iy0+1):
                if self.containsPoint(x,y,border):
                    Imask[y,x] = True

    def getImageMask(self, border=0.0):
        Imask = np.zeros(self.I.shape, dtype=bool)
        for x in range(int(math.floor(self.bounds[0]-border)),int(math.ceil(self.bounds[2]+border))+1):
            for y in range(int(math.floor(self.bounds[1]-border)),int(math.ceil(self.bounds[3]+border))+1):
                if self._containsPoint(x,y,border):
                    Imask[y,x] = True
        return Imask

    def getArea(self):
        total=0.0
        for i in range(0,len(self.thetas)):
            theta = (self.thetas[i+1] if i < len(self.thetas)-1 else 2*math.pi) -self.thetas[i]
            r1 = self.rs[i]
            r2 = self.rs[i+1] if i < len(self.thetas)-1 else self.rs[0]
            total += (r1*r2*math.sin(theta))/2
        return total

    '''def getAxes(self):
        if len(self.thetas)%2 != 0:
            raise RuntimeError("Axes can only be computed for contours with an even number of radii.")

        rmin=float("Inf")
        rmax=0.0
        for i in range(0,len(self.thetas)//2):
            r = self.rs[i]+self.rs[i+len(self.thetas)//2]
            if r > rmax:
                rmax = r
            if r < r:
        return (rmin,rmax)'''

    def containsPoint(self, x, y, border=0.0, showIntermediates=False):

        # Transform the point into cell coordinates and perform the check.
        return self._containsPoint(x-self.Ix0, y-self.Iy0, border, showIntermediates)

    def _containsPoint(self, x, y, border=0.0, showIntermediates=False):

        # See if the point falls outside our bounding box.
        if x < self.bounds[0]-border or x > self.bounds[2]+border or y < self.bounds[1]-border or y > self.bounds[3]+border:
            return False

        # Get polar coordinates and bins for the point.
        (r,theta,theta1,theta2)=self._findBinForPoint(x,y)

        # See if we are outside both rays.
        if r > self.rs[theta1]+border and r > self.rs[theta2]+border:
            return False

        # If this was the center point, return True.
        if r == 0.0:
            return True

        # Get the line connecting both rays.
        x1 = self.xs[theta1]+border*self.coss[theta1]
        y1 = self.ys[theta1]+border*self.sins[theta1]
        x2 = self.xs[theta2]+border*self.coss[theta2]
        y2 = self.ys[theta2]+border*self.sins[theta2]
        line1 = ((x1,y1),(x2,y2))

        # Get the line from the center through the point of interest.
        line2 = ((self.xc,self.yc),(x,y))

        # Get the intersection point.
        (xi,yi)=imaging.findIntersection(line1, line2)

        # Calculate the distance from the center to the intersection point.
        ri = math.sqrt((xi-self.xc)**2 + (yi-self.yc)**2)

        # If the point of interest is closer to the center than the intersection, it is inside the boundary.
        if r <= ri:
            return True
        else:
            return False


    def _findBinForPoint(self, x, y):

        # Calculate the polar coordinates of the point.
        theta = math.atan2(y-self.yc,x-self.xc)
        r = math.sqrt((x-self.xc)**2 + (y-self.yc)**2)
        if theta < 0.0: theta += 2*math.pi

        # Get the r/theta bin of the point.
        theta1 = int(theta/(self.thetas[1]-self.thetas[0]))
        theta2 = theta1+1
        if theta2 >= len(self.thetas): theta2 = 0

        return (r,theta,theta1,theta2)


    def distanceFromPoint(self, x, y):

        # Transform the point into cell coordinates and perform the check.
        return self._distanceFromPoint(x-self.Ix0, y-self.Iy0)

    def _distanceFromPoint(self, x, y):

        # Get polar coordinates and bins for the point.
        (r,theta,theta1,theta2)=self._findBinForPoint(x,y)

        # Get the line connecting both rays for this bin.
        line1 = ((self.xs[theta1],self.ys[theta1]),(self.xs[theta2],self.ys[theta2]))

        # Get the line from the center to the edge.
        line2 = ((self.xc,self.yc),(x,y))

        # Get the intersection point.
        (xi,yi)=imaging.findIntersection(line1, line2)

        # Get the distance from the center to the boundary.
        b = math.sqrt((xi-self.xc)**2 + (yi-self.yc)**2)

        return (r,(r-b))

    def distanceFromCenter(self, x, y):

        # Transform the point into cell coordinates and perform the call.
        return self.distanceFromCenter(x-self.Ix0, y-self.Iy0)

    def _distanceFromCenter(self, x, y):
        return math.sqrt((x-self.xc)**2 + (y-self.yc)**2)


    def setRs(self, rs):
        self.rs = rs
        self.xs = self.rs*self.coss+self.xc
        self.ys = self.rs*self.sins+self.yc
        self.bounds = (np.min(self.xs),np.min(self.ys),np.max(self.xs),np.max(self.ys))

    def setCenterPoint(self, xc, yc):

        # Create a local variable for the new rs.
        rs = np.zeros_like(self.rs)

        # Go through each pair of consecutive points on the outline.
        x1=self.xs[-1]-xc
        y1=self.ys[-1]-yc
        theta1 = math.atan2(y1, x1)
        if theta1 < 0.0: theta1 += 2*math.pi
        for i in range(0, len(self.rs)):

            # Get the angle for this point relative to the new center.
            x2=self.xs[i]-xc
            y2=self.ys[i]-yc
            theta2 = math.atan2(y2, x2)
            if theta2 < 0.0: theta2 += 2*math.pi

            if theta1 > theta2:
                thetaIndices = np.where(np.logical_or(self.thetas >= theta1, self.thetas < theta2))[0]
            else:
                thetaIndices = np.where(np.logical_and(self.thetas >= theta1, self.thetas < theta2))[0]

            # Go through every new theta that falls between these two points.
            for thetaIndex in thetaIndices:

                # Find the intersection of this theta with the line connecting the two points.
                (x,y)=imaging.findIntersection(((x1,y1),(x2,y2)), ((0.0,0.0),(self.coss[thetaIndex],self.sins[thetaIndex])))
                rs[thetaIndex] = math.sqrt(x**2 + y**2)

            # Update the previous x, y, and theta to be the current values.
            x1 = x2
            y1 = y2
            theta1 = theta2

        # Set the new values.
        self.xc = xc
        self.yc = yc
        self.rs = rs
        self.xs=self.rs*self.coss+self.xc
        self.ys=self.rs*self.sins+self.yc
        self.bounds = (np.min(self.xs),np.min(self.ys),np.max(self.xs),np.max(self.ys))

    def __str__(self):
        if id is None:
            return "xc=%0.2f,yc=%0.2f,#thetas=%d,<r>=%0.2f"%(self.xc,self.yc,len(self.thetas),np.mean(self.rs))
        else:
            return "%s,xc=%0.2f,yc=%0.2f,#thetas=%d,<r>=%0.2f"%(self.id.hex,self.xc,self.yc,len(self.thetas),np.mean(self.rs))



def refineCellInteractions(originalCells, border=10.0, k_radial=1.0, k_boundary=0.10, k_image=15.0, k_interact_repulse=0.5, k_interact_attract=0.05, tolerance=1e-2, showIntermediates=False):

    cells = copy.deepcopy(originalCells)

    # Find all of the overlapping pairs.
    for i in range(0,len(cells)-1):
        for j in range(i+1,len(cells)):
            if CellContour.doCellsOverlap(cells[i], cells[j], border=border):
                print "Optimzing interactions between %s and %s"%(cells[i].id,cells[j].id)
                (c1,c2)=optimizeYeastInteractionsModel([cells[i], cells[j]], 100.0, border, k_radial,k_boundary,k_image,k_interact_repulse, k_interact_attract, tolerance, showIntermediates=showIntermediates)
                cells[i] = c1
                cells[j] = c2

    return cells

def optimizeYeastInteractionsModel(guesses, rMax, border, k_radial,k_boundary,k_image,k_interact_repulse, k_interact_attract, tolerance, showIntermediates=False):

    # Create a copy of the cells.
    cells=(copy.deepcopy(guesses))

    # Find the regions where the cells overlap and create a list of them to be optimized.
    overlappingRadii=CellContour.getOverlapingRadii(cells[0], cells[1], border=border)

    # If there were no overlapping radii, we are done.
    if len(overlappingRadii) == 0:
        return cells

    # Get the original rs for the overlapping radii.
    x0=np.zeros((len(overlappingRadii),))
    radiiMap={}
    for (i,(ci,ri)) in enumerate(overlappingRadii):
        x0[i] = cells[ci].rs[ri]
        radiiMap[i] = (ci,ri)

    print "Optimizing %d radii"%(len(x0))
    print radiiMap
    print x0

    # Go through each cell and get some properties.
    Is=[None,None]
    Lrs=[None, None]
    xcircs=[None, None]
    ycircs=[None, None]
    Ixs=[None,None]
    Iys=[None,None]
    for ci in range(0,len(cells)):

        # Calculate a Guassian filtered image.
        Is[ci] = ndimage.gaussian_filter(cells[ci].I,4).astype(float)/256.0
        Imin =np.min(Is[ci])
        Is[ci] = (Is[ci]-Imin)/(1.0-Imin)

        # Calculate the image gradients.
        (Iy,Ix)=segment.calculateHVGradients(cells[ci].I)
        Ixs[ci] = Ix
        Iys[ci] = Iy

        # Create the radial second derivative matrix.
        Lrs[ci]=np.zeros((int(math.ceil(rMax)),len(cells[ci].thetas)),dtype=np.float64)
        Lrs[ci][0,:]=float("inf")
        Lrs[ci][-1,:]=float("inf")
        for i in range(1,Lrs[ci].shape[0]-1):
            for j in range(0,Lrs[ci].shape[1]):
                xm=float(i-1)*cells[ci].coss[j]+cells[ci].xc
                xp=float(i+1)*cells[ci].coss[j]+cells[ci].xc
                ym=float(i-1)*cells[ci].sins[j]+cells[ci].yc
                yp=float(i+1)*cells[ci].sins[j]+cells[ci].yc
                Lrs[ci][i,j]=-((imaging.bilinearInterpolate(Ix,xp,yp)-imaging.bilinearInterpolate(Ix,xm,ym))*cells[ci].coss[j] + (imaging.bilinearInterpolate(Iy,xp,yp)-imaging.bilinearInterpolate(Iy,xm,ym))*cells[ci].sins[j])

        # Fit the last cell outline to a circle for use in the boundary energy calculations.
        (xcirc,ycirc,rcirc)=fitCircleToYeastModel(cells[ci])
        xcircs[ci] = xcirc
        ycircs[ci] = ycirc

    # Optimize the joint radii.
    result=opt.minimize(yeastInteractionsModelFR, x0, args=(cells, radiiMap, Is, Lrs, xcircs, ycircs, k_radial,k_boundary,k_image,k_interact_repulse,k_interact_attract),method="BFGS",options={'maxiter':20000,'gtol':tolerance})
    if result.success is not True and result.message != "Desired error not necessarily achieved due to precision loss.":
        print result
        raise RuntimeError("Minimization FR did not converge.")
    print "DEBUG: Minimization Iteraction FR converged after %d evaluations: %0.3f"%(result.nfev,result.fun)

    # Set the new radii in the cells.
    for i in range(0, len(result.x)):
        cells[radiiMap[i][0]].rs[radiiMap[i][1]] = result.x[i]

    # Set the rs in each cell to update the other variables.
    cells[0].setRs(cells[0].rs)
    cells[1].setRs(cells[1].rs)

    if showIntermediates:
        for i,c in enumerate(cells):
            cprev=guesses[i]
            plt.figure()
            plt.subplot(2,4,1)
            plt.plot(cprev.thetas,cprev.rs,marker='x',color='r')
            plt.plot(c.thetas, c.rs, marker='x', color='b')
            plt.subplot(2,4,2)
            plt.imshow(c.I)
            plt.scatter(cprev.xc,cprev.yc,marker='x',color='r')
            plt.plot(cprev.xs,cprev.ys,marker='x',color='r')
            plt.scatter(c.xc,c.yc,marker='x',color='b')
            plt.plot(c.xs,c.ys,marker='x',color='b')
            #plt.scatter(xcirc,ycirc,marker='x',color='g')
            #plt.gcf().gca().add_artist(plt.Circle((xcirc,ycirc),rcirc,color='g',fill=False))
            for nc in cells:
                if nc.id != c.id:
                    plt.scatter(nc.xc+nc.Ix0-c.Ix0,nc.yc+nc.Iy0-c.Iy0,marker='x',color='k')
                    plt.plot(nc.xs+nc.Ix0-c.Ix0,nc.ys+nc.Iy0-c.Iy0,marker='x',color='k')
            plt.axis([c.bounds[0]-10,c.bounds[2]+10,c.bounds[3]+10,c.bounds[1]-10])
            plt.subplot(2,4,3)
            Ir=np.zeros((int(math.ceil(rMax)),len(c.thetas)),dtype=np.float64)
            for ii in range(0,Ir.shape[0]):
                for ij in range(0,Ir.shape[1]):
                    x=float(ii)*c.coss[ij]+c.xc
                    y=float(ii)*c.sins[ij]+c.yc
                    Ir[ii,ij] = imaging.bilinearInterpolate(Is[i],x,y)
            plt.imshow(Ir)
            plt.plot(range(0,len(c.rs)),c.rs,marker='x',color='b')
            plt.axis([0,Ir.shape[1],0,Ir.shape[0]])
            plt.subplot(2,4,4)
            plt.imshow(Lrs[i])
            plt.plot(range(0,len(c.rs)),c.rs,marker='x',color='b')
            plt.axis([0,Lrs[i].shape[1],0,Lrs[i].shape[0]])
            plt.subplot(2,4,5)
            if cells is not None:
                plt.plot(cprev.thetas, calculateYeastModelInteractionEnergies(cprev.rs, cprev.xs, cprev.ys, cprev, guesses, k_interact_repulse, k_interact_attract), marker='x', color='r')
                plt.plot(c.thetas, calculateYeastModelInteractionEnergies(c.rs, c.xs, c.ys, c, cells, k_interact_repulse, k_interact_attract), marker='x', color='b')
            plt.subplot(2,4,6)
            plt.plot(cprev.thetas, calculateYeastModelBoundaryEnergies(cprev.rs, cprev.xs, cprev.ys, xcircs[i], ycircs[i], k_boundary), marker='x', color='r')
            plt.plot(c.thetas, calculateYeastModelBoundaryEnergies(c.rs, c.xs, c.ys, xcircs[i], ycircs[i], k_boundary), marker='x', color='b')
            plt.subplot(2,4,7)
            plt.plot(cprev.thetas, calculateYeastModelImageEnergies(cprev.rs, cprev.xs, cprev.ys, Is[i], k_image), marker='x', color='r')
            plt.plot(c.thetas, calculateYeastModelImageEnergies(c.rs, c.xs, c.ys, Is[i], k_image), marker='x', color='b')
            plt.subplot(2,4,8)
            plt.plot(cprev.thetas, calculateYeastModelRadialGradientEnergiesLr(cprev.rs, Lrs[i], k_radial), marker='x', color='r')
            plt.plot(c.thetas, calculateYeastModelRadialGradientEnergiesLr(c.rs, Lrs[i], k_radial), marker='x', color='b')
            plt.show()


    return cells


def yeastInteractionsModelFR(rs, cellsOrig, radiiMap, Is, Lrs, xcircs, ycircs, k_radial, k_boundary, k_image, k_interact_repulse, k_interact_attract):

    # Make sure the radius is not out of range.
    if np.min(rs) < 2.0 or np.max(rs) > float(Lrs[0].shape[0]-1): return float("inf")

    # Make a copy of each cell.
    cells = copy.deepcopy(cellsOrig)

    # Update the cells with the new radii.
    for i in range(0, len(rs)):
        cells[radiiMap[i][0]].rs[radiiMap[i][1]] = rs[i]

    # Set the rs in each cell to update the other variables.
    cells[0].setRs(cells[0].rs)
    cells[1].setRs(cells[1].rs)

    # Calculate the total energy for the two cells.
    Etot=0.0
    for i,cell in enumerate(cells):

        # Add the radial gradient energy.
        Etot += np.sum(calculateYeastModelRadialGradientEnergiesLr(cell.rs, Lrs[i], k_radial))

        # Add the boundary energy.
        Etot += np.sum(calculateYeastModelBoundaryEnergies(cell.rs, cell.xs, cell.ys, xcircs[i], ycircs[i], k_boundary))

        # Add the image energy.
        Etot += np.sum(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, Is[i], k_image))

        # Add the interaction energy.
        Etot += np.sum(calculateYeastModelInteractionEnergies(cell.rs, cell.xs, cell.ys, cell, cells, k_interact_repulse, k_interact_attract))

    return Etot


def refineCellContour(guess, neighboringCells=None, border=10.0, k_radial=1.0, k_boundary=0.10, k_image=15.0, k_interact_repulse=0.5, k_interact_attract=0.0, tolerance=1e-2, showIntermediates=False):
    
    c=optimizeYeastModel(guess, rMax=100.0, possibleNeighboringCells=neighboringCells, border=border, k_radial=k_radial, k_boundary=k_boundary, k_image=k_image, k_interact_repulse=k_interact_repulse, k_interact_attract=k_interact_attract, tolerance=tolerance, showIntermediates=showIntermediates)
    
    if showIntermediates:
        plt.figure()
        plt.subplot(1,1,1)
        io.imshow(guess.I)
        plt.scatter(c.xc,c.yc,marker='x',color='g')
        plt.plot(c.xs,c.ys,marker='x',color='g')
        plt.scatter(guess.xc,guess.yc,marker='x',color='r')
        plt.plot(guess.xs,guess.ys,marker='x',color='r')
        plt.axis([0,guess.I.shape[1],guess.I.shape[0],0])
        io.show()
        
    return c


def optimizeYeastModel(guess, rMax, possibleNeighboringCells, border, k_radial, k_boundary, k_image, k_interact_repulse, k_interact_attract, tolerance, showIntermediates=False):

    c=copy.deepcopy(guess)

    # Find any neighboring cells that could possibly be near the cell of interest.
    neighboringCells=None
    if possibleNeighboringCells is not None:
        neighboringCells=[]
        for possibleNeighboringCell in possibleNeighboringCells:
            if CellContour.doCellsOverlap(c, possibleNeighboringCell, border):
                neighboringCells.append(possibleNeighboringCell)

    # Calculate a Guassian filtered image.
    I = ndimage.gaussian_filter(c.I, 4).astype(float)/256.0
    Imin =np.min(I)
    I = (I-Imin)/(1.0-Imin)

    # Calculate the image gradients.
    (Iy,Ix)=segment.calculateHVGradients(c.I)

    for optRound in range(0,3):

        cprev = copy.deepcopy(c)

        # Fit the last cell outline to a circle for use in the boundary energy calculations.
        (xcirc,ycirc,rcirc)=fitCircleToYeastModel(c)

        stage = optRound%2
        if stage == 0:

            # Create the radial second derivative matrix.
            Lr=np.zeros((int(math.ceil(rMax)),len(c.thetas)),dtype=np.float64)
            Lr[0,:]=float("inf")
            Lr[-1,:]=float("inf")
            for i in range(1,Lr.shape[0]-1):
                for j in range(0,Lr.shape[1]):
                    xm=float(i-1)*c.coss[j]+c.xc
                    xp=float(i+1)*c.coss[j]+c.xc
                    ym=float(i-1)*c.sins[j]+c.yc
                    yp=float(i+1)*c.sins[j]+c.yc
                    Lr[i,j]=-((imaging.bilinearInterpolate(Ix,xp,yp)-imaging.bilinearInterpolate(Ix,xm,ym))*c.coss[j] + (imaging.bilinearInterpolate(Iy,xp,yp)-imaging.bilinearInterpolate(Iy,xm,ym))*c.sins[j])

            x0 = c.rs
            result=opt.minimize(yeastModelFR, x0, args=(I,Lr,c,xcirc,ycirc,neighboringCells,k_radial,k_boundary,k_image,k_interact_repulse, k_interact_attract),method="BFGS",options={'maxiter':20000,'gtol':tolerance})
            #result=opt.minimize(yeastModelFR, x0, args=(I,Lr,c,xcirc,ycirc,k_radial,k_boundary,k_image),method="CG",options={'maxiter':20000,'eps':1.0,'gtol':1e-2})
            #result=opt.minimize(yeastModelFR, x0, args=(I,Lr,c,xcirc,ycirc,k_radial,k_boundary,k_image),method="Nelder-Mead",options={'maxfev':20000,'xtol':1e-1,'ftol':1e-1})
            #minimizer_kwargs={"args":(I,Lr,c,xcirc,ycirc,0.0,k_boundary,0.0), "method":"Nelder-Mead", "options":{'maxfev':20000,'xtol':1e-1,'ftol':1e-1}}
            #minimizer_kwargs={"args":(I,Lr,c,xcirc,ycirc,0.0,k_boundary,0.0), "method":"BFGS"}
            #minimizer_kwargs={"args":(I,Lr,c,xcirc,ycirc,0.0,k_boundary,0.0)}
            #result=opt.basinhopping(yeastModelFR, x0, minimizer_kwargs=minimizer_kwargs, disp=True, niter=1)
            if result.success is not True and result.message != "Desired error not necessarily achieved due to precision loss.":
                print result
                raise RuntimeError("Minimization FR did not converge.")
            print "DEBUG: Minimization FR converged after %d evaluations: %0.3f"%(result.nfev,result.fun)
            c.setRs(result.x)

            if False:
                printCellEnergies(I, Lr, c, xcirc, ycirc, neighboringCells, k_radial, k_boundary, k_image, k_interact_repulse, k_interact_attract)


        elif stage == 1:
            # Optimize the center of the cell.
            x0 = np.asanyarray([c.xc,c.yc],dtype=float)
            result = opt.minimize(yeastModelFC, x0, args=(Ix, Iy, c, k_radial),method="BFGS",options={'maxiter':20000,'gtol':tolerance})
            #result = opt.minimize(yeastModelFC, x0, args=(I,Ix,Iy,c,xcirc,ycirc,k_radial,k_boundary,k_image),method="Nelder-Mead",options={'maxfev':20000,'xtol':1e-1,'ftol':1e-1})
            #minimizer_kwargs={"args":(I,Ix,Iy,c,xcirc,ycirc,k_radial,k_boundary,k_image)}
            #result=opt.basinhopping(yeastModelFC, x0, minimizer_kwargs=minimizer_kwargs, disp=True, niter=10)
            if result.success is not True and result.message != "Desired error not necessarily achieved due to precision loss.":
                print result
                raise RuntimeError("Minimization FC did not converge.")
            print "DEBUG: Minimization FC converged after %d evaluations: %0.3f"%(result.nfev,result.fun)
            c.setCenterPoint(result.x[0], result.x[1])

        if showIntermediates:
            plt.figure()
            plt.subplot(2,4,1)
            plt.plot(cprev.thetas,cprev.rs,marker='x',color='r')
            plt.plot(c.thetas, c.rs, marker='x', color='b')
            plt.subplot(2,4,2)
            plt.imshow(c.I)
            plt.scatter(cprev.xc,cprev.yc,marker='x',color='r')
            plt.plot(cprev.xs,cprev.ys,marker='x',color='r')
            plt.scatter(c.xc,c.yc,marker='x',color='b')
            plt.plot(c.xs,c.ys,marker='x',color='b')
            plt.scatter(xcirc,ycirc,marker='x',color='g')
            plt.gcf().gca().add_artist(plt.Circle((xcirc,ycirc),rcirc,color='g',fill=False))
            if neighboringCells is not None:
                for nc in neighboringCells:
                    if nc.id != c.id:
                        plt.scatter(nc.xc+nc.Ix0-c.Ix0,nc.yc+nc.Iy0-c.Iy0,marker='x',color='k')
                        plt.plot(nc.xs+nc.Ix0-c.Ix0,nc.ys+nc.Iy0-c.Iy0,marker='x',color='k')
            plt.axis([c.bounds[0]-10,c.bounds[2]+10,c.bounds[3]+10,c.bounds[1]-10])
            #plt.axis([0,c.I.shape[1],c.I.shape[0],0])
            plt.subplot(2,4,3)
            Ir=np.zeros((int(math.ceil(rMax)),len(c.thetas)),dtype=np.float64)
            for i in range(0,Ir.shape[0]):
                for j in range(0,Ir.shape[1]):
                    x=float(i)*c.coss[j]+c.xc
                    y=float(i)*c.sins[j]+c.yc
                    Ir[i,j] = imaging.bilinearInterpolate(I,x,y)
            plt.imshow(Ir)
            plt.plot(range(0,len(c.rs)),c.rs,marker='x',color='b')
            plt.axis([0,Ir.shape[1],0,Ir.shape[0]])
            plt.subplot(2,4,4)
            Lr2=np.zeros((int(math.ceil(rMax)),len(c.thetas)),dtype=np.float64)
            Lr2[0,:]=float("inf")
            Lr2[-1,:]=float("inf")
            for i in range(1,Lr2.shape[0]-1):
                for j in range(0,Lr2.shape[1]):
                    xm=float(i-1)*c.coss[j]+c.xc
                    xp=float(i+1)*c.coss[j]+c.xc
                    ym=float(i-1)*c.sins[j]+c.yc
                    yp=float(i+1)*c.sins[j]+c.yc
                    Lr2[i,j]=-((imaging.bilinearInterpolate(Ix,xp,yp)-imaging.bilinearInterpolate(Ix,xm,ym))*c.coss[j] + (imaging.bilinearInterpolate(Iy,xp,yp)-imaging.bilinearInterpolate(Iy,xm,ym))*c.sins[j])
            plt.imshow(Lr2)
            plt.plot(range(0,len(c.rs)),c.rs,marker='x',color='b')
            plt.axis([0,Lr2.shape[1],0,Lr2.shape[0]])

            plt.subplot(2,4,5)
            if neighboringCells is not None:
                Eint = calculateYeastModelInteractionEnergies(c.rs, c.xs, c.ys, c, neighboringCells, k_interact_repulse, k_interact_attract)
                plt.plot(c.thetas, Eint, marker='x', color='b')
            plt.subplot(2,4,6)
            Ebound = calculateYeastModelBoundaryEnergies(c.rs, c.xs, c.ys, xcirc, ycirc, k_boundary)
            plt.plot(c.thetas, Ebound, marker='x', color='b')
            plt.subplot(2,4,7)
            Eimage = calculateYeastModelImageEnergies(c.rs, c.xs, c.ys, I, k_image)
            plt.plot(c.thetas, Eimage, marker='x', color='b')
            plt.subplot(2,4,8)
            Egrad = calculateYeastModelRadialGradientEnergies(c.rs, c.xc, c.yc, c.sins, c.coss, Ix, Iy, k_radial)
            plt.plot(c.thetas, Egrad, marker='x', color='b')
            plt.show()

        # dxs=np.arange(-20,21)
        # dys=np.arange(-20,21)
        # E=np.zeros((len(dxs),len(dys)),dtype=float)
        # for i in range(0,len(dxs)):
        #     for j in range(0,len(dys)):
        #         E[i,j]=yeastModelFC((c.xc+dxs[i],c.yc+dys[j]), I, Ix, Iy, rMax, c)-result.fun
        # print E.min()
        # print E.max()
        # plt.figure()
        # io.imshow(E)
        # plt.axis([0,E.shape[1],0,E.shape[0]])
        # io.show()

    return c

def printCellEnergies(I, Lr, c, xcirc, ycirc, neighboringCells, k_radial, k_boundary, k_image, k_interact_repulse, k_interact_attract):
    original_options = np.get_printoptions()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, formatter={'float': lambda x: "%10.3e,"%(x)})
    cellString = "Cell (new)" if c.id is None else "Cell (%s)"%(c.id)
    print "DEBUG: %s index:             "%(cellString),np.arange(0.0,len(c.rs), dtype=float)
    print "DEBUG: %s radii:             "%(cellString),c.rs
    print "DEBUG: %s xs:                "%(cellString),c.xs
    print "DEBUG: %s ys:                "%(cellString),c.ys
    print "DEBUG: %s gradient energies: "%(cellString),calculateYeastModelRadialGradientEnergiesLr(c.rs, Lr, k_radial)
    print "DEBUG: %s boundary energies: "%(cellString),calculateYeastModelBoundaryEnergies(c.rs, c.xs, c.ys, xcirc, ycirc, k_boundary)
    print "DEBUG: %s image energies:    "%(cellString),calculateYeastModelImageEnergies(c.rs, c.xs, c.ys, I, k_image)
    if neighboringCells is not None:
        print "DEBUG: %s interact energies: "%(cellString),calculateYeastModelInteractionEnergies(c.rs, c.xs, c.ys, c, neighboringCells, k_interact_repulse, k_interact_attract)
    np.set_printoptions(**original_options)


def yeastModelFR(rs, I, Lr, c, xcirc, ycirc, neighboringCells, k_radial, k_boundary, k_image, k_interact_repulse, k_interact_attract):

    # Calculate the x and y coordinates.
    xs=rs*c.coss+c.xc
    ys=rs*c.sins+c.yc
    
    # Make sure the radius is not out of range.
    if np.min(rs) < 2.0 or np.max(rs) > float(Lr.shape[0]-1): return float("inf")

    # Calculate the energy.
    Etot=0.0

    # Add the radial gradient energy.
    Etot += np.sum(calculateYeastModelRadialGradientEnergiesLr(rs, Lr, k_radial))

    # Add the boundary energy.
    Etot += np.sum(calculateYeastModelBoundaryEnergies(rs, xs, ys, xcirc, ycirc, k_boundary))

    # Add the image energy.
    Etot += np.sum(calculateYeastModelImageEnergies(rs, xs, ys, I, k_image))

    # Add the interaction energy.
    if neighboringCells is not None:
        Etot += np.sum(calculateYeastModelInteractionEnergies(rs, xs, ys, c, neighboringCells, k_interact_repulse, k_interact_attract))

    return Etot

def yeastModelFC(x, Ix, Iy, c, k_radial):

    # Calculate the x and y coordinates.
    xc=x[0]
    yc=x[1]

    c2 = copy.deepcopy(c)

    try:
        c2.setCenterPoint(xc, yc)
    except RuntimeError:
        print "Exception setting center point to %f,%f from %f,%f"%(xc,yc,c.xc,c.yc)
        return float("inf")

    # Make sure the radius is not out of range.
    if np.min(c2.rs) < 2.0: return float("inf")

    # Calculate the energy.
    Etot=0.0

    # Add the radial gradient energy.
    Etot += np.sum(calculateYeastModelRadialGradientEnergies(c2.rs, c2.xc, c2.yc, c2.sins, c2.coss, Ix, Iy, k_radial))

    return Etot

def calculateYeastModelRadialGradientEnergiesLr(rs, Lr, k_radial):

    E = np.zeros((len(rs),), dtype=float)

    try:
        for i in range(0,len(rs)):

            # Add energy for the radial second derivative.
            E[i] = imaging.linearInterpolateY(Lr,i,rs[i])

    except (RuntimeError,IndexError):
        print "Exception calculating radial gradient energy for r %d=%0.2f"%(i,rs[i])
        E[:]=float("inf")
        return E

    # Scale the energies.
    E *= k_radial

    return E

def calculateYeastModelRadialGradientEnergies(rs, xc, yc, sins, coss, Ix, Iy, k_radial):

    E = np.zeros((len(rs),), dtype=float)

    try:
        for i in range(0,len(rs)):

            # Add energy for the radial second derivative.
            xm=(rs[i]-1.0)*coss[i]+xc
            xp=(rs[i]+1.0)*coss[i]+xc
            ym=(rs[i]-1.0)*sins[i]+yc
            yp=(rs[i]+1.0)*sins[i]+yc
            E[i] = -((imaging.bilinearInterpolate(Ix,xp,yp)-imaging.bilinearInterpolate(Ix,xm,ym))*coss[i] + (imaging.bilinearInterpolate(Iy,xp,yp)-imaging.bilinearInterpolate(Iy,xm,ym))*sins[i])
    except (RuntimeError,IndexError):
        print "Exception calculating radial gradient energy for r %d=%0.2f"%(i,rs[i])
        E[:]=float("inf")
        return E

    # Scale the energies.
    E *= k_radial

    return E

def calculateYeastModelBoundaryEnergies(rs, xs, ys, xcirc, ycirc, k_boundary):

    E = np.zeros((len(rs),), dtype=float)

    # Calculate the thetas with respect to the reference circle.
    thetas = np.arctan2(ys-ycirc,xs-xcirc)
    thetas[thetas<0] += 2*math.pi

    try:
        # Go through each boundary point.
        for i in range(0,len(rs)):

            # Calculate the previous and next index.
            h = i-1 if i > 0 else len(rs)-1
            j = i+1 if i < len(rs)-1 else 0

            # Calculate the inside angle for this point relative to the previous and next points.
            a1=math.atan2(ys[i]-ys[h], xs[i]-xs[h])
            if a1 < 0.0: a1+=2*math.pi
            a2=math.atan2(ys[j]-ys[i], xs[j]-xs[i])
            if a2 < 0.0: a2+=2*math.pi
            if a2-a1 < -math.pi: a2 += 2*math.pi
            elif a2-a1 > math.pi: a2 -= 2*math.pi
            phi=math.pi-(a2-a1)

            # Calculate the inside angle for this point if all three fell on the specified circle.
            a1 = thetas[h]
            a2 = thetas[j]
            if a2 < a1: a2 += 2*math.pi
            phi0 = math.pi-((a2-a1)/2)

            # Add energy for the convexity of this segment.
            if phi <= phi0:
                E[i] = ((phi0-phi))
            else:
                E[i] = ((phi-phi0))
    except (RuntimeError,IndexError):
        print "Exception calculating boundary energy for point %d,%d for r %d=%0.2f"%(xs[i],ys[i],i,rs[i])
        E[:]=float("inf")
        return E

    # Scale the energies.
    E *= k_boundary

    return E

def calculateYeastModelImageEnergies(rs, xs, ys, I, k_image):

    E = np.zeros((len(rs),), dtype=float)

    try:
        # Go through each boundary point.
        for i in range(0,len(rs)):

            # Add energy for the image.
            E[i] = imaging.bilinearInterpolate(I,xs[i],ys[i])/256.0

    except (RuntimeError,IndexError):
        print "Exception calculating image energy for point %d,%d for r %d=%0.2f"%(xs[i],ys[i],i,rs[i])
        E[:]=float("inf")
        return E

    # Scale the energies.
    E *= k_image

    return E

def calculateYeastModelInteractionEnergies(rs, xs, ys, cell, neighboringCells, k_interact_repulse, k_interact_attract):

    E = np.zeros((len(rs),), dtype=float)
    xs = np.copy(xs)+cell.Ix0
    ys = np.copy(ys)+cell.Iy0
    xc = cell.xc + cell.Ix0
    yc = cell.yc + cell.Iy0

    # Go through each boundary point.
    for i in range(0,len(rs)):
        for neighboringCell in neighboringCells:
            if neighboringCell.id != cell.id:
                (r,db)=neighboringCell.distanceFromPoint(xs[i], ys[i])
                r0 = r-db
                kr=k_interact_repulse
                ka=k_interact_attract
                kdr=2.5
                kda=0.5
                if r <= r0:
                    E[i] = (-(ka+kr)*np.exp((1/kdr)*(r-r0))+kr)
                elif r <= r0+5*kda:
                    E[i] = (-ka*np.exp(-(1/kda)*(r-r0)))
    return E

def fitCircleToYeastModel(cell):

    x0=np.asarray([cell.xc,cell.yc,np.mean(cell.rs)], dtype=float)
    result=opt.minimize(fitCircleToYeastModelF, x0, args=(cell,), method="Nelder-Mead",options={'maxfev':20000,'xtol':1e-2,'ftol':1e-2})
    if result.success is not True:
        print result
        raise RuntimeError("fitCircleToYeastModel minimization did not converge.")
    return (result.x[0],result.x[1],result.x[2])


def fitCircleToYeastModelF(x, cell):

    (xcirc,ycirc,r)=x
    dx = cell.xs-xcirc
    dy = cell.ys-ycirc
    return np.sum((np.sqrt(dx**2+dy**2)-r)**2)


def validateCellContour(cell, showIntermediates=False, **validateOptions):

    # Check each of the filter options.
    isvalid=True
    newCell=cell.id is None
    for key, value in validateOptions.iteritems():
        if key == "area" and not validate_area(cell, value, newCell, showIntermediates):
            isvalid=False
        if key == "interiorIntensity" and not validate_interiorIntensity(cell, value, newCell, showIntermediates):
            isvalid=False
        if key == "borderIntensity" and not validate_borderIntensity(cell, value, newCell, showIntermediates):
            isvalid=False
        if key == "interiorBorderIntensityRatio" and not validate_interiorBorderIntensityRatio(cell, value, newCell, showIntermediates):
            isvalid=False
        if key == "radius" and not validate_radius(cell, value, newCell, showIntermediates):
            isvalid=False
        if key == "gradientEnergy" and not validate_gradientEnergy(cell, value, newCell, showIntermediates):
            isvalid=False

    return isvalid

def validate_area(cell, validRange, newCell, showIntermediates=False):
    value=cell.getArea()
    if newCell:
        isValid = validRange[0] <= value <= validRange[1]
    else:
        isValid = value <= validRange[1]
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has an area of %0.4f, range %0.4f-%0.4f"%(prefix,cellString,value,validRange[0],validRange[1])
    return isValid

def validate_interiorIntensity(cell, validRange, newCell, showIntermediates=False):
    Imask = cell.getImageMask()
    indices = np.where(Imask == True)
    values = cell.I[indices]
    m = np.mean(values)/255.0
    s = (np.std(values)/255.0)/m
    isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has an interior intensity of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,cellString,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

def validate_borderIntensity(cell, validRange, newCell, showIntermediates=False):
    m=np.mean(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, cell.I, 1.0))
    s=np.std(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, cell.I, 1.0))/m
    isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has a border intensity of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,cellString,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

def validate_interiorBorderIntensityRatio(cell, validRange, newCell, showIntermediates=False):
    Imask = cell.getImageMask()
    indices = np.where(Imask == True)
    values = cell.I[indices]
    value1 = np.mean(values)/255.0
    value2=np.sum(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, cell.I, 1.0))/float(len(cell.rs))
    value=value1/value2
    isValid = validRange[0] <= value <= validRange[1]
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has an interior to border ratio of %0.4f, range %0.4f-%0.4f"%(prefix,cellString,value,validRange[0],validRange[1])
    return isValid

def validate_radius(cell, validRange, newCell, showIntermediates=False):
    xcom = np.mean(cell.xs)
    ycom = np.mean(cell.ys)
    dxs = cell.xs - xcom
    dys = cell.ys - ycom
    rs = np.sqrt(dxs**2+dys**2)
    m = np.mean(rs)
    s = np.std(rs)/m
    if newCell:
        isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    else:
        isValid = m <= validRange[1] and s < validRange[2]
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has a radius of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,cellString,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

def validate_gradientEnergy(cell, validRange, newCell, showIntermediates=False):
    (Iy,Ix)=segment.calculateHVGradients(cell.I)
    Es=calculateYeastModelRadialGradientEnergies(cell.rs, cell.xc, cell.yc, cell.sins, cell.coss, Ix, Iy, 1.0)
    m=np.mean(Es)
    s=np.std(Es)/abs(m)
    if newCell:
        isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    else:
        isValid = validRange[0]*1.1 <= m <= 0 and s < validRange[2]*1.1
    if showIntermediates or not isValid:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has a gradient energy of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,cellString,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

