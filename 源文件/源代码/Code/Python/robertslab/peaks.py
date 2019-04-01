from __future__ import division
import math
import numpy as np

def findLocalMaxima1D(data):
    
    # Get the indices of the values in sorted order.
    sortedIndices=np.argsort(data, axis=None)
    
    # Create an empty list of peaks.
    peaks=[]
    
    # Go through the values in reverse order and put them into peaks.
    i=len(sortedIndices)-1
    x=sortedIndices[i]
    v=data[x]
    while i >= 0 and v > 0:

        # See if the points falls into any of the existing peaks.
        matches=[]
        for p in peaks:
            if p.encompassesPoint(x,v):
                matches.append(p)
                
        # If we didn't find any peaks, create a new one
        if len(matches) == 0:
            peaks.append(Peak1D(x,v))
            
        # Otherwise, if we found only one match, add the point.
        elif len(matches) == 1:
            matches[0].addPoint(x,v)
        
        # Otherwise, we found multiple possible matches.
        else:
            # Add it to the closest peak.
            closestPeak=matches[0]
            for p in matches[1:-1]:
                if p.distanceFromCenter(x) < closestPeak.distanceFromCenter(x):
                    closestPeak = p
            closestPeak.addPoint(x,v)
            
            # See if any peaks need to be combined.
            p1=matches[0]
            j=1
            while j<len(matches):
                p2=matches[j]
                if (p1.distanceFromCenter(p2.xc)) < (p1.rechargeRadius+p2.rechargeRadius):
                    p1.combineWith(p2)
                    peaks.remove(p2)
                    matches.remove(p2)
                else:
                    j+=1
        
        # Refresh the indices.
        i-=1
        x=sortedIndices[i]
        v=data[x]

    # Combine any overlapping peaks.
    for i in range(0,len(peaks)-1):
        for j in range(i+1,len(peaks)):
            p1=peaks[i]
            p2=peaks[j]
            if not p1.isEmpty() and not p2.isEmpty():
                
                r=p1.distanceFromCenter(p2.xc)
                
                # If the recharge radii overlap, combine them.
                if r < (p1.rechargeRadius+p2.rechargeRadius):
                    p1.combineWith(p2)
                    p2.deletePoints()
                
                # If the radii overlap, combine them if p2 is not high enough.
                elif r < (p1.coverageRadius+p2.coverageRadius) and p2.maxValue < p1.maxValue*0.3:
                    p1.combineWith(p2)
                    p2.deletePoints()
                
    
    # Remove any empty peaks.
    peaks2=[]
    for i in range(0,len(peaks)):
        if not peaks[i].isEmpty():
            peaks2.append(peaks[i])
        
    return peaks2



def findLocalMaxima2D(data):
    
    # Get the indices of the values in sorted order.
    sortedIndices=np.argsort(data, axis=None)
    
    # Create an empty list of peaks.
    peaks=[]
    
    # Go through the values in reverse order and put them into peaks.
    i=len(sortedIndices)-1
    (x,y)=_index1Dto2D(sortedIndices[i],data.shape)
    v=data[y,x]
    while i >= 0 and v > 0:
        
        # See if the points falls into any of the existing peaks.
        matches=[]
        for p in peaks:
            if p.encompassesPoint(x,y,v):
                matches.append(p)
                
        # If we didn't find any peaks, create a new one
        if len(matches) == 0:
            peaks.append(Peak(x,y,v))
            #print "Added new peak %d,%d,%0.2f"%(x,y,v)
            
        # Otherwise, if we found only one match, add the point.
        elif len(matches) == 1:
            matches[0].addPoint(x,y,v)
            #print "Added to existing peak %d,%d,%0.2f"%(x,y,v)
        
        # Otherwise, we found multiple possible matches.
        else:
            # Add it to the closest peak.
            #print("%d found %d matches"%(i,len(matches)))
            closestPeak=matches[0]
            for p in matches[1:-1]:
                if p.distanceFromCenter(x,y) < closestPeak.distanceFromCenter(x,y):
                    closestPeak = p
            closestPeak.addPoint(x,y,v)
            
            # See if any peaks need to be combined.
            #print("Combining:%d,%d"%(len(matches),len(peaks)))
            p1=matches[0]
            j=1
            while j<len(matches):
                p2=matches[j]
                if (p1.distanceFromCenter(p2.xc,p2.yc)) < (p1.rechargeRadius+p2.rechargeRadius):
                    p1.combineWith(p2)
                    peaks.remove(p2)
                    matches.remove(p2)
                    #print("Combined %s with %s"%(str(p1),str(p2)))
                else:
                    j+=1
            #print("Combined:%d,%d"%(len(matches),len(peaks)))
        
        # Refresh the indices.
        i-=1
        (x,y)=_index1Dto2D(sortedIndices[i],data.shape)
        v=data[y,x]
        
    # Combine any overlapping peaks.
    #print("Combining")
    for i in range(0,len(peaks)-1):
        for j in range(i+1,len(peaks)):
            p1=peaks[i]
            p2=peaks[j]
            if not p1.isEmpty() and not p2.isEmpty():
                
                r=p1.distanceFromCenter(p2.xc,p2.yc)
                #print("%d -> %d = %0.2f   %s %s"%(i,j,r,str(p1),str(p2)))
                
                # If the recharge radii overlap, combine them.
                if r < (p1.rechargeRadius+p2.rechargeRadius):
                    p1.combineWith(p2)
                    p2.deletePoints()
                    #print("y1")
                
                # If the radii overlap, combine them if p2 is not high enough.
                elif r < (p1.coverageRadius+p2.coverageRadius) and p2.maxValue < p1.maxValue*0.3:
                    p1.combineWith(p2)
                    p2.deletePoints()
                    #print("y2")
                
    
    # Remove any empty peaks.
    peaks2=[]
    for i in range(0,len(peaks)):
        if not peaks[i].isEmpty():
            peaks2.append(peaks[i])
        
    return peaks2

################################################################################
# Functions called primarily from within the package.
################################################################################

class Peak1D:
    def __init__(self,x,v):
        self.xc=float(x)
        self.centerWeight=float(v)
        self.maxValue=v
        self.coverageRadius=0
        self.rechargeRadius=0
        self.points=[]
        self.points.append((x,v))
            
    def __str__(self):
        if self.isEmpty(): return "peak:empty"
        return "peak:xc=%0.2f,maxValue=%0.2f,radius=%0.2f:%0.2f,count=%d"%(self.xc,self.maxValue,self.rechargeRadius,self.coverageRadius,len(self.points))
    
    def deletePoints(self):
        self.xc=None
        self.centerWeight=None
        self.maxValue=None
        self.coverageRadius=None
        self.rechargeRadius=None
        self.points=[]
        
    def isEmpty(self):
        return len(self.points) == 0
        
    def distanceFromCenter(self, x):
        
        if self.isEmpty(): return None
            
        return (x-self.xc)

    def encompassesPoint(self, x, v):
        
        if self.isEmpty(): return False
            
        # See if the point falls within the peak.
        return (x-self.xc) <= self.coverageRadius+5
    
    def combineWith(self, p2):
        for point in p2.points:
            self.addPoint(point[0],point[1])

    def addPoint(self, x, v):
        
        self.points.append((x,v))

        # Recalculate the weighted center.
        self.xc=(self.xc*self.centerWeight+x*v)/(self.centerWeight+v)
        self.centerWeight+=v
        
        # See if the radius has changed.
        r=(x-self.xc)
        self.coverageRadius=max(r,self.coverageRadius)
        if v >= self.maxValue*0.5:
            self.rechargeRadius=max(r,self.rechargeRadius)


def _index1Dto2D(index, shape):
    return (index%shape[1],index//shape[1])
    
class Peak:
    def __init__(self,x,y,v):
        self.xc=float(x)
        self.yc=float(y)
        self.centerWeight=float(v)
        self.xmin=x
        self.xmax=x
        self.ymin=y
        self.ymax=y
        self.maxValue=v
        self.coverageRadius=0
        self.rechargeRadius=0
        self.points=[]
        self.points.append((x,y,v))
            
    def __str__(self):
        if self.isEmpty(): return "peak:empty"
        return "peak:xc=%0.2f,yc=%0.2f,maxValue=%0.2f,radius=%0.2f:%0.2f,count=%d"%(self.xc,self.yc,self.maxValue,self.rechargeRadius,self.coverageRadius,len(self.points))
    
    def deletePoints(self):
        self.xc=None
        self.yc=None
        self.centerWeight=None
        self.maxValue=None
        self.coverageRadius=None
        self.rechargeRadius=None
        self.points=[]
        
    def isEmpty(self):
        return len(self.points) == 0
        
    def distanceFromCenter(self, x, y):
        
        if self.isEmpty(): return None
            
        return math.sqrt((x-self.xc)**2+(y-self.yc)**2)

    def encompassesPoint(self, x, y, v):
        
        if self.isEmpty(): return False
            
        # See if the point falls within the peak.
        return math.sqrt((x-self.xc)**2+(y-self.yc)**2) <= self.coverageRadius+5
    
    def combineWith(self, p2):
        for point in p2.points:
            self.addPoint(point[0],point[1],point[2])

    def addPoint(self, x, y, v):
        
        self.points.append((x,y,v))

        # Recalculate the weighted center.
        self.xc=(self.xc*self.centerWeight+x*v)/(self.centerWeight+v)
        self.yc=(self.yc*self.centerWeight+y*v)/(self.centerWeight+v)
        self.centerWeight+=v
        
        # See if the radius has changed.
        r=math.sqrt((x-self.xc)**2+(y-self.yc)**2)
        self.coverageRadius=max(r,self.coverageRadius)
        if v >= self.maxValue*0.5:
            self.rechargeRadius=max(r,self.rechargeRadius)
        
        
        
        
        
