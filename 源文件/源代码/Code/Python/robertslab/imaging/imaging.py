from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from skimage import io

import robertslab.peaks as peaks

def linearInterpolateY(im, x, y):

    y0 = np.floor(y).astype(int)
    y1 = y0
    if y1 < im.shape[0]-1:
        y1 += 1

    Ia = im[y0,x]
    Ib = im[y1,x]

    wa = (y1-y)
    wb = (y-y0)

    return wa*Ia + wb*Ib    

def linearInterpolateX(im, x, y):

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    Ia = im[y,x0]
    Ib = im[y,x1]

    wa = (x1-x)
    wb = (x-x0)

    return wa*Ia + wb*Ib    

def bilinearInterpolate(im, x, y):

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    Ia = im[y0,x0]
    Ib = im[y1,x0]
    Ic = im[y0,x1]
    Id = im[y1,x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def findIntersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise RuntimeError("lines do not intersect: %f,%f:%f,%f %f,%f:%f,%f"%(line1[0]+line1[1]+line2[0]+line2[1]))

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y

def calculateBackgroundStats(im, showIntermediates=False):

    # Histogram the data.
    (imC,imX)=np.histogram(im,bins=np.arange(0,256,1.0))

    # Find the highest peak, which corresponds to the background.
    maxima=peaks.findLocalMaxima1D(imC)
    backgroundPeak=maxima[0]
    for i in range(1,len(maxima)):
        if maxima[i].maxValue > backgroundPeak.maxValue:
            backgroundPeak=maxima[i]

    # Find the boundaries of the background peak.
    for i in range(int(backgroundPeak.xc), -1, -1):
        if imC[i] < backgroundPeak.maxValue*0.05:
            bgMin=i+1
            break
    for i in range(int(backgroundPeak.xc), len(imC)):
        if imC[i] < backgroundPeak.maxValue*0.05:
            bgMax=i-1
            break

    # Calculate the background statistics.
    D=imC[bgMin:bgMax].astype(np.float64)/np.sum(imC[bgMin:bgMax])
    B=imX[bgMin:bgMax]
    bgMean=np.sum(B*D)
    bgStd=math.sqrt(np.sum(((B-bgMean)**2)*D))

    if showIntermediates:
        plt.figure()
        plt.subplot(2,1,1)
        io.imshow(im)
        plt.subplot(2,1,2)
        plt.plot(imX[:-1],imC)
        plt.axvline(bgMean, color='r')
        plt.axvline(bgMean+3*bgStd, color='g')
        plt.axvline(bgMean-3*bgStd, color='g')
        x = range(bgMin,bgMax,1)
        plt.plot(x, norm.pdf(x,bgMean,bgStd)*np.sum(imC[bgMin:bgMax]),color='r')
        io.show()

    return (bgMean,bgStd)


def extractImageRegion(I, x1s, y1s, x2s, y2s, defaultValue):

    # Figure out the boundaries of the region to extract for processing.
    x1d=0
    y1d=0
    x2d=x2s-x1s
    y2d=y2s-y1s

    # Create an empty cell image.
    Iextract=np.zeros((y2s-y1s,x2s-x1s),dtype=I.dtype)+defaultValue

    # If the cell region is off the edge of the image, adjust the boundaries
    if x1s < 0:
        x1d+=0-x1s
        x1s=0
    if x2s > I.shape[1]:
        x2d-=x2s-I.shape[1]
        x2s=I.shape[1]
    if y1s < 0:
        y1d+=0-y1s
        y1s=0
    if y2s > I.shape[0]:
        y2d-=y2s-I.shape[0]
        y2s=I.shape[0]

    # Copy the cell image into the buffer.
    Iextract[y1d:y2d,x1d:x2d] = I[y1s:y2s,x1s:x2s]

    return Iextract
