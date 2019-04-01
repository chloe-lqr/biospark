from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from skimage import io, filter

class Grid:
    def __init__(self,orig=None):
        if (orig != None):
            self.firstVerticalXIntercept=orig.firstVerticalXIntercept
            self.firstHorizontalYIntercept=orig.firstHorizontalYIntercept
            self.rotationAngle=orig.rotationAngle
            self.spacing=orig.spacing
            self.firstVerticalTriple=orig.firstVerticalTriple
            self.firstHorizontalTriple=orig.firstHorizontalTriple
            self.tripleSpacing=orig.tripleSpacing
        else:
            self.firstVerticalXIntercept=0
            self.firstHorizontalYIntercept=0
            self.rotationAngle=0.5
            self.spacing=205
            self.firstVerticalTriple=0
            self.firstHorizontalTriple=0
            self.tripleSpacing=20.0
            
    def __str__(self):
        return "grid:x=%0.2f,y=%0.2f,angle=%0.2f,spacing=%0.2f,tripleV=%d,tripleH=%d,tripleSpacing=%0.2f"%(self.firstVerticalXIntercept,self.firstHorizontalYIntercept,self.rotationAngle,self.spacing,self.firstVerticalTriple,self.firstHorizontalTriple,self.tripleSpacing)
    

def detectGrid(im, showIntermediates=False):
    
    (g,score)=estimateInitialGrid(im, showIntermediates=showIntermediates)
    print("Initial estimate: %s, score:%0.2f"%(str(g),score))
    
    (g,score)=refineGrid(im, g, score, showIntermediates=showIntermediates)
    print("Final estimate: %s, score:%0.2f"%(str(g),score))
    
    return (g,score)
    
    
def estimateInitialGrid(im, showIntermediates=False):
    
    # Take the horizontal gradient, log-normalize it, and then threshold it.
    imH=filter.hsobel(im)
    imH[imH<1e-4]=1e-4
    limH=np.log10(imH)
    limH+=-np.min(limH)
    limH/=np.max(limH)
    threshold=np.mean(limH)+np.std(limH)
    limH[limH>threshold]=1
    limH[limH<1]=0
    
    # Take the vertical gradient, log-normalize it, and then threshold it.
    imV=filter.vsobel(im)
    imV[imV<1e-4]=1e-4
    limV=np.log10(imV)
    limV+=-np.min(limV)
    limV/=np.max(limV)
    threshold=np.mean(limV)+np.std(limV)
    limV[limV>threshold]=1
    limV[limV<1]=0

    # Find the horizontal density peaks in the left quarter of the image.
    dH1=np.sum(limH[:,0:im.shape[1]/4],axis=1)
    (peaksH1,widthsH1)=findMaxima(dH1,100,10,50)
    
    # Find the horizontal density peaks in the right quarter of the image.
    dH2=np.sum(limH[:,im.shape[1]*3/4:im.shape[1]],axis=1)
    (peaksH2,widthsH2)=findMaxima(dH2,100,10,50)

    # Find the vertical density peaks in the top quarter of the image.
    dV1=np.sum(limV[0:im.shape[0]/4,:],axis=0)
    (peaksV1,widthsV1)=findMaxima(dV1,100,10,50)
    
    # Find the vertical density peaks in the bottom quarter of the image.
    dV2=np.sum(limV[im.shape[0]*3/4:im.shape[0],:],axis=0)
    (peaksV2,widthsV2)=findMaxima(dV2,100,10,50)
    
    if showIntermediates:
        plt.figure()
        plt.subplot(4,2,1)
        io.imshow(imH)
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(4,2,2)
        io.imshow(imV)
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(4,2,3)
        io.imshow(limH)
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(4,2,4)
        io.imshow(limV)
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(4,2,5)
        plt.plot(dH1)
        for i in range(0,len(peaksH1)):
            plt.axvline(peaksH1[i], color='r')
            plt.axvline(peaksH1[i]-widthsH1[i]/2, color='g')
            plt.axvline(peaksH1[i]+widthsH1[i]/2, color='g')
        plt.subplot(4,2,6)
        plt.plot(dH2)
        for i in range(0,len(peaksH2)):
            plt.axvline(peaksH2[i], color='r')
            plt.axvline(peaksH2[i]-widthsH2[i]/2, color='g')
            plt.axvline(peaksH2[i]+widthsH2[i]/2, color='g')
        plt.subplot(4,2,7)
        plt.plot(dV1)
        for i in range(0,len(peaksV1)):
            plt.axvline(peaksV1[i], color='r')
            plt.axvline(peaksV1[i]-widthsV1[i]/2, color='g')
            plt.axvline(peaksV1[i]+widthsV1[i]/2, color='g')
        plt.subplot(4,2,8)
        plt.plot(dV2)
        for i in range(0,len(peaksV2)):
            plt.axvline(peaksV2[i], color='r')
            plt.axvline(peaksV2[i]-widthsV2[i]/2, color='g')
            plt.axvline(peaksV2[i]+widthsV2[i]/2, color='g')
        io.show()
    
    # Find the horizontal lines.
    linesH=[]
    widthsH=[]
    for i in range(0,len(peaksH1)):
        for j in range(0,len(peaksH2)):
            if abs(peaksH1[i]-peaksH2[j]) < 25:
                linesH.append((peaksH1[i],peaksH2[j]))
                widthsH.append((widthsH1[i]+widthsH2[j])/2)
                break
                
    # Find the vertical lines.
    linesV=[]
    widthsV=[]
    for i in range(0,len(peaksV1)):
        for j in range(0,len(peaksV2)):
            if abs(peaksV1[i]-peaksV2[j]) < 25:
                linesV.append((peaksV1[i],peaksV2[j]))
                widthsV.append((widthsV1[i]+widthsV2[j])/2)
                break

    # Create the initial grid estimate.
    g=Grid()

    # Save the intercepts.
    g.firstVerticalXIntercept=linesV[0][0]
    g.firstHorizontalYIntercept=linesH[0][0]

    # Calculate the average rotation.
    total=0.0
    count=0
    for line in linesV:
        total+=math.degrees(math.atan(abs(line[0]-line[1])/float(im.shape[0])))
        count+=1
    for line in linesH:
        total+=math.degrees(math.atan(abs(line[0]-line[1])/float(im.shape[1])))
        count+=1
    g.rotationAngle=total/count
    
    # Calculate the average spacing.
    if len(linesV) >= 2 or len(linesH) >= 2:
        total=0.0
        count=0
        for i in range(0,len(linesV)-1):
            total+=linesV[i+1][0]-linesV[i][0]
            count+=1
        for i in range(0,len(linesH)-1):
            total+=linesH[i+1][0]-linesH[i][0]
            count+=1
        g.spacing=(total/count)*math.cos(math.radians(g.rotationAngle))
    
    # Figure out which line is the first vertical triple.
    g.firstVerticalTriple=0
    avgWidth=sum(widthsV)/len(widthsV)
    for i in range(0,len(linesV)):
        if widthsV[i] > avgWidth*1.5:
            g.firstVerticalTriple=i
            break
    
    # Figure out which line is the first vertical triple.
    g.firstHorizontalTriple=0
    avgWidth=sum(widthsH)/len(widthsH)
    for i in range(0,len(linesH)):
        if widthsH[i] > avgWidth*1.5:
            g.firstHorizontalTriple=i
            break
            
    # Show the intermediate results, if asked to.
    if showIntermediates:
        plt.figure()
        plt.subplot(2,2,1)
        io.imshow(im)
        for i in range(0,len(peaksH1)):
            plt.plot((0,im.shape[1]/4),(peaksH1[i],peaksH1[i]), 'g-')
        for i in range(0,len(peaksH2)):
            plt.plot((im.shape[1]*3/4,im.shape[1]),(peaksH2[i],peaksH2[i]), 'g-')
        for i in range(0,len(peaksV1)):
            plt.plot((peaksV1[i],peaksV1[i]),(0,im.shape[0]/4), 'g-')
        for i in range(0,len(peaksV2)):
            plt.plot((peaksV2[i],peaksV2[i]),(im.shape[0]*3/4,im.shape[0]), 'g-')
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(2,2,2)
        io.imshow(im)
        for i in range(0,len(linesH)):
            plt.plot((0,im.shape[1]),(linesH[i][0],linesH[i][1]), 'g-')
        for i in range(0,len(linesV)):
            plt.plot((linesV[i][0],linesV[i][1]),(0,im.shape[0]), 'g-')
        plt.axis([0,im.shape[1],im.shape[0],0])
        plt.subplot(2,2,3)
        gridIm=np.zeros(im.shape,dtype=np.uint8)
        drawGrid(gridIm,g)
        gridIm[gridIm>0]=255
        colorIm=np.zeros(im.shape+(3,),dtype=np.uint8)
        colorIm[:,:,0]=im
        colorIm[:,:,1]=np.maximum(im,gridIm)
        colorIm[:,:,2]=im
        io.imshow(colorIm)
        plt.axis([0,im.shape[1],im.shape[0],0])
        io.show()
        
    score=calculateOverlayScore(im, g)
    return (g,score)
        
        
def refineGrid(im, g, score, showIntermediates=False):
    
    maxTrial=600
    for trial in range(0,maxTrial):
        
        # Copy the previous best guess.
        tmp_g=Grid(g)
        
        # Make a new guess.
        trialType=random.randint(0,6)
        if trialType == 0:
            tmp_g.firstVerticalXIntercept+=random.randint(int(round(-10.0*((maxTrial-trial)/maxTrial))),int(round(10.0*((maxTrial-trial)/maxTrial))))
        elif trialType == 1:
            tmp_g.firstHorizontalYIntercept+=random.randint(int(round(-10.0*((maxTrial-trial)/maxTrial))),int(round(10.0*((maxTrial-trial)/maxTrial))))
        elif trialType == 2 or trialType == 4:
            tmp_g.rotationAngle+=random.gauss(0.0,0.1*((maxTrial-trial)/maxTrial))
        elif trialType == 3 or trialType == 4:
            tmp_g.spacing+=random.gauss(0.0,5*((maxTrial-trial)/maxTrial))
        elif trialType == 5:
            if random.randint(0,1) == 0:
                tmp_g.firstVerticalTriple=random.randint(0,3)
            else:
                tmp_g.firstHorizontalTriple=random.randint(0,3)
        elif trialType == 6:
            firstTime=True
            while firstTime or tmp_g.tripleSpacing < 15.0:
                tmp_g.tripleSpacing+=random.gauss(0.0,3*((maxTrial-trial)/maxTrial))
                firstTime=False
            
        # Calculate the new score.
        tmp_score=calculateOverlayScore(im, tmp_g, showIntermediates=False)
        
        # If this score was better, save it as the new best guess.
        if tmp_score > score:
            print("New estimate at trial %d with trial type %d: %s, score:%0.2f"%(trial+1,trialType,str(tmp_g),tmp_score))
            score=tmp_score
            g=tmp_g
        
    return (g,score)


################################################################################
# Functions called primarily from within the package.
################################################################################
        
def calculateOverlayScore(im,g,showIntermediates=False):
    
    # Get the grid values for each pixel of the image.
    gridIm=np.zeros(im.shape,dtype=np.uint8)
    drawGrid(gridIm,g)
    overlap=im*(gridIm/255.0)
    
    # Show the intermediate results, if asked to.
    if showIntermediates:
        plt.figure()
        io.imshow(overlap)
        plt.axis([0,im.shape[1],im.shape[0],0])
        io.show()
    
    # Calculate the overlap between the grid and the image pixel intensities.
    return np.sum(overlap)
    
        
def findMaxima(data, yThreshold, minHighRunLength, minLowRunLength):
    
    # Thresholded data.
    thresholdedData=data>=yThreshold
    
    # Calculate the runs above the threshold.
    aboveThesholdRuns=[]
    runStart=0
    runState=thresholdedData[runStart]
    for i in range(1,data.shape[0]):
        if thresholdedData[i] != runState:
            if runState:
                aboveThesholdRuns.append((runStart,i-1))
            runStart=i
            runState=thresholdedData[i]
    
    # Remove any gaps between runs that are not long enough.
    i=0
    while i<(len(aboveThesholdRuns)-1):
        if (aboveThesholdRuns[i+1][0]-aboveThesholdRuns[i][1]-1) < minLowRunLength:
            aboveThesholdRuns[i]=(aboveThesholdRuns[i][0],aboveThesholdRuns[i+1][1])
            del aboveThesholdRuns[i+1]
        else:
            i+=1
        
    # Remove runs that are not long enough.
    i=0
    while i<(len(aboveThesholdRuns)-1):
        if (aboveThesholdRuns[i][1]-aboveThesholdRuns[i][0]+1) < minHighRunLength:
            del aboveThesholdRuns[i]
        else:
            i+=1

    maxima=[]
    widths=[]
    for run in aboveThesholdRuns:
        maxima.append((run[0]+run[1])/2)
        widths.append((run[1]-run[0]+1))
    return (maxima,widths)


def drawGrid(im, g):
    x=g.firstVerticalXIntercept
    x2dx=math.tan(math.radians(g.rotationAngle))*im.shape[0]
    i=0
    while x<im.shape[1]:
        drawLine(im,(x,0),(x-x2dx,im.shape[0]))
        if i%4 == g.firstVerticalTriple:
            drawLine(im,(x-g.tripleSpacing,0),(x-x2dx-g.tripleSpacing,im.shape[0]))
            drawLine(im,(x+g.tripleSpacing,0),(x-x2dx+g.tripleSpacing,im.shape[0]))
        x+=g.spacing/math.cos(math.radians(g.rotationAngle))
        i+=1
    y=g.firstHorizontalYIntercept
    y2dy=math.tan(math.radians(g.rotationAngle))*im.shape[1]
    i=0
    while y<im.shape[0]:
        drawLine(im,(0,y),(im.shape[1],y+y2dy))
        if i%4 == g.firstHorizontalTriple:
            drawLine(im,(0,y-g.tripleSpacing),(im.shape[1],y+y2dy-g.tripleSpacing))
            drawLine(im,(0,y+g.tripleSpacing),(im.shape[1],y+y2dy+g.tripleSpacing))
        y+=g.spacing/math.cos(math.radians(g.rotationAngle))
        i+=1

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
        img[xy[1]][xy[0]]=int(alpha*255)

