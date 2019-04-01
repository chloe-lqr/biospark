import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from skimage import io

import robertslab.imaging.imaging as imaging
import robertslab.imaging.yeast.segment as segment

class Bead:

    def __init__(self, xc, yc, r):
        self.xc = xc
        self.yc = yc
        self.r = r

    def fillImageMask(self, Imask, border=0.0):
        for x in range(int(math.floor(self.xc-self.r-border)),int(math.ceil(self.xc+self.r+border))):
            for y in range(int(math.floor(self.yc-self.r-border)),int(math.ceil(self.yc+self.r+border))):
                if self.containsPoint(x,y,border):
                    Imask[y,x] = True

    def containsPoint(self, x, y, border=0.0):
        if x < self.xc-self.r-border or x > self.xc+self.r+border or y < self.yc-self.r-border or y > self.yc+self.r+border:
            return False
        return ((x-self.xc)**2+(y-self.yc)**2) <= (self.r**2+border)

    def __str__(self):
        return "xc=%0.2f,yc=%0.2f,r=%0.2f"%(self.xc,self.yc,self.r)

def findBead(I, G, Gthresh, Gtheta, validateOptions=None, showIntermediates=False):

    beads = findBeads(I, G, Gthresh, Gtheta, validateOptions, showIntermediates)

    return beads[0]

def findBeads(I, G, Gthresh, Gtheta, validateOptions=None, showIntermediates=False):

    # Calculate the center of the reference bead.
    centers = findPotentialBeadCenters(I, G, Gthresh, Gtheta, showIntermediates=showIntermediates)

    # Go through all of the centers and fit to a bead model.
    beads = []
    for (bx,by,br) in centers:

        # Extract a region around the position of the reference bead from the current frame.
        imBeadX=int(bx)-300
        imBeadY=int(by)-300
        imBead=imaging.extractImageRegion(I,imBeadX,imBeadY,imBeadX+600,imBeadY+600,0)

        # Fit a circle to the bead.
        (xc,yc,r)=fitCircleModel(imBead, 300, 300, br, showIntermediates=showIntermediates)

        # Create the bead model, translating the center back to image coordinates.
        bead = Bead(xc+imBeadX,imBeadY+yc,r)

        # If the bead is valid, add it to the list.
        if validateOptions is None:
            beads.append(bead)
        elif validateBead(I, bead, validateOptions, showIntermediates):
            beads.append(bead)

    if showIntermediates:
        plt.figure()
        io.imshow(I)
        for bead in beads:
            plt.scatter(bead.xc,bead.yc,marker='x',color='g')
            plt.gcf().gca().add_artist(plt.Circle((bead.xc,bead.yc),bead.r,color='g',fill=False))
        plt.axis([0,I.shape[1],I.shape[0],0])
        io.show()

    return beads

def findPotentialBeadCenters(I, G, Gthresh, Gtheta, showIntermediates=False):

    # Calculate the Hough transform.
    (hough,houghThresh)=segment.calculateHoughTransform(G, Gthresh, Gtheta, 20, direction=1, showIntermediates=showIntermediates)

    # Estimate the cluster centers from the Hough transform.
    centers=segment.estimateCenters(hough, houghThresh, showIntermediates=showIntermediates)

    return centers

def fitCircleModel(im, xGuess, yGuess, rGuess, showIntermediates=False):

    x0=np.asarray([xGuess,yGuess,rGuess], dtype=float)
    result=opt.minimize(circleModelF, x0, args=(im,),method="CG",options={'eps':1.0,'gtol':1e-2})
    return (result.x[0],result.x[1],result.x[2])


def circleModelF(s, im):

    xc=s[0]
    yc=s[1]
    r=s[2]

    thetas=np.arange(0.0, 2*np.pi, 2*np.pi/50.0)
    xs=r*np.cos(thetas)+xc
    ys=r*np.sin(thetas)+yc
    E=0.0
    for i in range(0, len(xs)):
        E += imaging.bilinearInterpolate(im,xs[i],ys[i])

    E/=(50.0*255.0)
    return E

def validateBead(I, bead, validateOptions, showIntermediates=False):

    # Check each of the filter options.
    isvalid=True
    for key, value in validateOptions.iteritems():
        if key == "interiorIntensity" and not validate_interiorIntensity(I, bead, value, showIntermediates):
            isvalid=False
        if key == "radialSymmetry" and not validate_radialSymmetry(I, bead, value, showIntermediates):
            isvalid=False
        if key == "borderIntensity" and not validate_borderIntensity(I, bead, value, showIntermediates):
            isvalid=False

    return isvalid

def validate_interiorIntensity(I, bead, validRange, showIntermediates=False):
    Imask = np.zeros(I.shape, dtype=bool)
    bead.fillImageMask(Imask)
    indices = np.where(Imask == True)
    values = I[indices]
    m = np.mean(values)/255.0
    s = (np.std(values)/255.0)/m
    isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    if showIntermediates:
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s bead has an interior intensity of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

def validate_radialSymmetry(I, bead, validRange, showIntermediates=False):

    # Fill in the radial intensity matrix.
    thetas=np.arange(0.0, 2*np.pi, 2*np.pi/20.0)
    rs=np.arange(0.0, bead.r, bead.r/10.0)
    R=np.zeros((len(rs),len(thetas)))
    for i in range(0,len(rs)):
        xs=rs[i]*np.cos(thetas)+bead.xc
        ys=rs[i]*np.sin(thetas)+bead.yc
        for j in range(0,len(thetas)):
            R[i,j]=imaging.bilinearInterpolate(I,xs[j],ys[j])

    m = np.mean(R,axis=1)/255.0
    s = (np.std(R,axis=1)/255.0)/m
    isValid = np.max(s) <= validRange[0]
    if showIntermediates:
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s bead has an radial symmetry of %0.4f, required <= %0.4f"%(prefix,np.max(s),validRange[0])
    return isValid


def validate_borderIntensity(I, bead, validRange, showIntermediates=False):
    m=np.mean(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, cell.I, 1.0))
    s=np.std(calculateYeastModelImageEnergies(cell.rs, cell.xs, cell.ys, cell.I, 1.0))/m
    isValid = validRange[0] <= m <= validRange[1] and s < validRange[2]
    if showIntermediates:
        cellString = "Cell (new)" if newCell else "Cell (%s)"%(cell.id)
        prefix = "DEBUG: Validation passed:" if isValid else "DEBUG Validation failed:"
        print "%s %s has a border intensity of %0.4f (std=%0.4f), range %0.4f-%0.4f (std=%0.4f)"%(prefix,cellString,m,s,validRange[0],validRange[1],validRange[2])
    return isValid

