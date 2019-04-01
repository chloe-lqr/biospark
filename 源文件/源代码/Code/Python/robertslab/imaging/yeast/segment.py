from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import scipy
from scipy.ndimage import convolve
import scipy.cluster.vq as cluster
from skimage import io,draw,img_as_float
from distutils.version import LooseVersion
import skimage
if LooseVersion(skimage.__version__) < LooseVersion("0.11.0"):
    from skimage import filter as filters
else:
    from skimage import filters as filters

from robertslab import peaks

def alignImages(im1, im2, maxOffset=5):
    
    bestScore=float("inf")
    bestOffset=(0,0)
    for dx in range(-maxOffset,maxOffset+1):
        for dy in range(-maxOffset,maxOffset+1):
            score=0.0
            for x in range(0,im2.shape[1]-abs(dx)):
                for y in range(0,im2.shape[0]-abs(dy)):
                    if dx >= 0 and dy >= 0:
                        score += abs((float(im2[y,x])-float(im1[y+dy,x+dx])))
                    elif dx < 0 and dy >= 0:
                        score += abs((float(im2[y,x-dx])-float(im1[y+dy,x])))
                    elif dx >= 0 and dy < 0:
                        score += abs((float(im2[y-dy,x])-float(im1[y,x+dx])))
                    elif dx < 0 and dy < 0:
                        score += abs((float(im2[y-dy,x-dx])-float(im1[y,x])))
            score /= (im2.shape[1]-abs(dx))*(im2.shape[0]-abs(dy))
            #print "Score for %d,%d was %0.4e (best=%0.4e)"%(dx,dy,score,bestScore)
            if score < bestScore:
                bestOffset=(dx,dy)
                bestScore=score
                
    return bestOffset

def detectCells(im, maxRadius=50, showIntermediates=False):
    
    # Calculate the gradient of the image.
    (grad,gradThresh,gradDirection)=calculateGradient(im, showIntermediates=showIntermediates)
    
    # Calculate the Hough transform.
    (hough,houghThresh)=calculateHoughTransform(grad, gradThresh, gradDirection, maxRadius, showIntermediates=showIntermediates)
    
    # Estimate the cluster centers from the Hough transform.
    centers=estimateCenters(hough, houghThresh, showIntermediates=showIntermediates)
    
    if showIntermediates:
        plt.figure()
        imNorm=128*(im/np.max(im))
        #print imNorm.shape
        #print("IM min=%0.2f max=%0.2f mean=%0.2f"%(np.min(imNorm),np.max(imNorm),np.mean(imNorm)))        
        #print("H min=%0.2f max=%0.2f mean=%0.2f"%(np.min(hough),np.max(hough),np.mean(hough)))        
        houghNorm=(128*(hough/np.max(hough)))
        combinedIm=np.zeros(im.shape+(3,),dtype=np.uint8)
        combinedIm[:,:,0]=imNorm
        combinedIm[:,:,1]=imNorm+houghNorm
        combinedIm[:,:,2]=imNorm
        io.imshow(combinedIm)
        
        for i in range(0,len(centers)):
            c=centers[i]
            plt.scatter(c[0],c[1],marker='x',color='g')
            plt.gcf().gca().add_artist(plt.Circle((c[0],c[1]),c[2],color='r',fill=False))
            plt.gcf().gca().text(c[0],c[1],str(i),fontsize=12,color='r')
        plt.axis([0,im.shape[1],0,im.shape[0]])
        io.show()
        
    # Find the candidate centers using a Hough transform with a circle.
    
    
def calculateHVGradients(im, gradientType="sobel", cutoff="auto", showIntermediates=False):
    
    # Calculate the gradient.
    if gradientType == "sobel":
        gradH=hsobelExt(im)
        gradV=vsobelExt(im)
    else:
        print("Unknown gradient type %s"%(gradientType))
        return
        
    return (gradH,gradV)

def calculateGradient(im, gradientType="sobel", cutoff="auto", showIntermediates=False):
    
    # Calculate the gradient.
    if gradientType == "sobel":
        gradH=hsobelExt(im)
        gradV=vsobelExt(im)
        #print("H gradient min=%0.2f max=%0.2f mean=%0.2f"%(np.min(gradH),np.max(gradH),np.mean(gradH)))
        #print("V gradient min=%0.2f max=%0.2f mean=%0.2f"%(np.min(gradV),np.max(gradV),np.mean(gradV)))
        grad=np.sqrt(np.power(gradH,2)+np.power(gradV,2))
        grad=(grad/np.max(grad))
        gradDirection=np.arctan2(gradH,gradV)
        #print("D gradient min=%0.2f max=%0.2f mean=%0.2f"%(np.min(gradDirection),np.max(gradDirection),np.mean(gradDirection)))
    elif gradientType == "scharr":
        grad=filters.scharr(im)
    else:
        print("Unknown gradient type %s"%(gradientType))
        return
        
    # Determine the gradient cutoff.
    (imC,imX)=np.histogram(im,bins=100)
    (gradC,gradX)=np.histogram(grad,bins=100)
    
    # Find the threshold.
    if cutoff == "auto":
        gradCutoff=np.mean(grad)+8*np.std(grad)
    else:
        gradCutoff=cutoff

    # Create the threshold image.
    gradThresh=np.zeros(grad.shape,dtype=np.uint8)
    gradThresh[grad>gradCutoff]=1

    if showIntermediates:
        plt.figure()
        plt.subplot(2,4,1)
        io.imshow(im)
        plt.subplot(2,4,2)
        io.imshow(grad)
        plt.subplot(2,4,3)
        io.imshow(gradH)
        plt.subplot(2,4,4)
        io.imshow(gradV)
        plt.subplot(2,4,5)
        plt.plot(gradX[:-1],np.log10(gradC))
        plt.axvline(gradCutoff, color='r')
        plt.subplot(2,4,6)
        io.imshow(gradThresh)
        plt.subplot(2,4,7)
        io.imshow(gradDirection*gradThresh)
        io.show()
    
    if cutoff == "auto":
        return (grad,gradThresh,gradDirection,gradCutoff)
    else:
        return (grad,gradThresh,gradDirection)

def calculateHoughTransform(gradFull, gradThreshFull, gradDirectionFull, maxRadius, direction=0, exclusionMask=None, showIntermediates=False):

    gradCosFull=np.cos(gradDirectionFull)*gradThreshFull
    gradSinFull=np.sin(gradDirectionFull)*gradThreshFull

    if exclusionMask is not None:
        inclusionMask = (exclusionMask!=True)
        grad = gradFull*inclusionMask
        gradThresh = gradThreshFull*inclusionMask
        gradCos = gradCosFull*inclusionMask
        gradSin = gradSinFull*inclusionMask
    else:
        grad = gradFull
        gradThresh = gradThreshFull
        gradCos = gradCosFull
        gradSin = gradSinFull

    accumulatorFull=np.zeros(grad.shape, dtype=float)
    if showIntermediates:
        accumulatorView=np.zeros(grad.shape, dtype=float)
    
    for x in range(0,grad.shape[1]):
        for y in range(0,grad.shape[0]):
            if gradThresh[y,x] > 0:
                dx=gradCos[y,x]*maxRadius
                dy=gradSin[y,x]*maxRadius
                if direction == 0 or direction == 2:
                    rr,cc = draw.line(y, x, int(round(y-dy)), int(round(x-dx)))
                    for i in range(0,len(rr)):
                        if rr[i] >= 0 and rr[i] < grad.shape[0] and cc[i] >= 0 and cc[i] < grad.shape[1]:
                            accumulatorFull[rr[i],cc[i]] += grad[y,x]
                            if showIntermediates and random.randint(0,20)==0:
                                accumulatorView[rr[i],cc[i]] = 1.0
                if direction == 1 or direction == 2:
                    rr,cc = draw.line(y, x, int(round(y+dy)), int(round(x+dx)))
                    for i in range(0,len(rr)):
                        if rr[i] >= 0 and rr[i] < grad.shape[0] and cc[i] >= 0 and cc[i] < grad.shape[1]:
                            accumulatorFull[rr[i],cc[i]] += grad[y,x]
                            if showIntermediates and random.randint(0,20)==0:
                                accumulatorView[rr[i],cc[i]] = 1.0

    if exclusionMask is not None:
        accumulator = accumulatorFull*inclusionMask
        if showIntermediates:
            accumulatorView *= inclusionMask
    else:
        accumulator = accumulatorFull

    #threshold=np.mean(accumulator)+(5*np.std(accumulator))
    threshold=1.5
    accumulatorThresh=np.zeros(accumulator.shape,dtype=np.uint8)
    accumulatorThresh[accumulator>threshold]=1


    
    if showIntermediates:
        plt.figure()
        plt.subplot(3,2,1)
        plt.imshow(gradFull)
        plt.subplot(3,2,2)
        if exclusionMask is not None:
            plt.imshow(exclusionMask)
        plt.subplot(3,2,3)
        plt.imshow(gradThreshFull)
        plt.subplot(3,2,4)
        plt.imshow(gradThresh)
        plt.subplot(3,2,5)
        plt.imshow(accumulatorFull)
        plt.subplot(3,2,6)
        plt.imshow(accumulator)
        #plt.subplot(3,2,6)
        #(accumulatorC,accumulatorX)=np.histogram(accumulator,bins=100)
        #plt.plot(accumulatorX[:-1],np.log10(accumulatorC))
        #plt.axvline(threshold, color='r')
        plt.show()
    
    return (accumulator,accumulatorThresh)


def estimateCenters(hough, houghThresh, exclusionMask=None, showIntermediates=False):
    
    # Find the local clusters.
    peakList=peaks.findLocalMaxima2D(hough*houghThresh)

    # Extract a list of the centers.
    centers=[]
    for p in peakList:
        if p.coverageRadius >= 10 and len(p.points) >= 50:
            centers.append((p.xc,p.yc,p.coverageRadius))
    
    if showIntermediates:
        plt.figure()
        plt.subplot(1,2,1)
        io.imshow(houghThresh)
        i=0
        for p in peakList:
            if p.coverageRadius >= 10 and len(p.points) >= 50:
                plt.scatter(p.xc,p.yc,marker='x',color='g')
                plt.gcf().gca().add_artist(plt.Circle((p.xc,p.yc),p.coverageRadius,color='g',fill=False))
                plt.gcf().gca().add_artist(plt.Circle((p.xc,p.yc),p.rechargeRadius,color='g',fill=False))
                plt.gcf().gca().text(p.xc,p.yc,str(i),fontsize=12,color='g')
            else:
                plt.scatter(p.xc,p.yc,marker='x',color='r')
                plt.gcf().gca().add_artist(plt.Circle((p.xc,p.yc),p.coverageRadius,color='r',fill=False))
                plt.gcf().gca().add_artist(plt.Circle((p.xc,p.yc),p.rechargeRadius,color='r',fill=False))
                plt.gcf().gca().text(p.xc,p.yc,str(i),fontsize=12,color='r')
            i+=1
        plt.axis([0,houghThresh.shape[1],houghThresh.shape[0],0])
        plt.subplot(1,2,2)
        io.imshow(houghThresh)
        for (xc,yc,r) in centers:
            plt.scatter(xc,yc,marker='x',color='g')
            plt.gcf().gca().add_artist(plt.Circle((xc,yc),r,color='g',fill=False))
        io.show()
    
    return centers
    
    '''
    # Find all of the non-zero elements.
    (r,c)=(houghThresh>0).nonzero()
    
    print r.shape
    print c.shape
    d=np.zeros((r.shape[0],2))
    d[:,0]=r
    d[:,1]=c
    
    for i in range(2,40):
    
        (clusters,score) = cluster.kmeans(d,i)
            '''
    


    
HSOBEL_WEIGHTS_3 = np.array([[ 1, 2, 1],
                           [ 0, 0, 0],
                           [-1,-2,-1]]) / 4.0
VSOBEL_WEIGHTS_3 = HSOBEL_WEIGHTS_3.T

HSOBEL_WEIGHTS_7 = np.array([[ 3, 4, 5, 6, 5, 4, 3],
                             [ 2, 3, 4, 5, 4, 3, 2],
                             [ 1, 2, 3, 4, 3, 2, 1],
                             [ 0, 0, 0, 0, 0, 0, 0],
                             [-1,-2,-3,-4,-3,-2,-1],
                             [-2,-3,-4,-5,-4,-3,-2],
                             [-3,-4,-5,-6,-5,-4,-3]]) / 69.0
VSOBEL_WEIGHTS_7 = HSOBEL_WEIGHTS_7.T

def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        mask = binary_erosion(mask, EROSION_SELEM, border_value=0)
        return result * mask

def hsobelExt(image, mask=None):
    image = img_as_float(image)
    #result = np.abs(convolve(image, HSOBEL_WEIGHTS))
    result = convolve(image, HSOBEL_WEIGHTS_7)
    return _mask_filter_result(result, mask)


def vsobelExt(image, mask=None):
    image = img_as_float(image)
    #result = np.abs(convolve(image, VSOBEL_WEIGHTS))
    result = convolve(image, VSOBEL_WEIGHTS_7)
    return _mask_filter_result(result, mask)

    
    
    
    
"""Draws an anti-aliased line in img from p1 to p2 using Xiaolin Wu's line algorithm."""
def drawLine(img, p1, p2):
    color=0xFF
    x1, y1, x2, y2 = p1 + p2
    dx, dy = x2-x1, y2-y1
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px,py), (py,px))[steep]
 
    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1
 
    grad = dy/dx
    intery = y1 + _rfpart(x1) * grad
 
    xstart = draw_endpoint(img, p(*p1), color, grad) + 1
    xend = draw_endpoint(img, p(*p2), color, grad)
 
    for x in range(xstart, xend):
        y = int(intery)
        putpixel(img, p(x, y), color, _rfpart(intery))
        putpixel(img, p(x, y+1), color, _fpart(intery))
        intery += grad
 
def draw_endpoint(img, pt, color, grad):
    x, y = pt
    xend = round(x)
    yend = y + grad * (xend - x)
    xgap = _rfpart(x + 0.5)
    px, py = int(xend), int(yend)
    putpixel(img, (px, py), color, _rfpart(yend) * xgap)
    putpixel(img, (px, py+1), color, _fpart(yend) * xgap)
    return px
        
def _fpart(x):
    return x - int(x)
 
def _rfpart(x):
    return 1 - _fpart(x)
 
def putpixel(img, xy, color, alpha=1):
    if xy[0] >= 0 and xy[1] >= 0 and xy[0] < img.shape[1] and xy[1] < img.shape[0]:
        img[xy[1]][xy[0]]=int(alpha*3.0)
    
    
    
    
    
    
    
    
    
    
    
    
