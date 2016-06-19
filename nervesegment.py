# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:30:26 2016

@author: pravinth
"""

import glob
import cv2 as cv
import numpy as np

imgShapeY = 420
imgShapeX = 580
reshapeFactor = 0.25


def loadTrainData():
    pathToData = './Data/train/'
    mask_files = sorted(glob.glob(pathToData + '*_mask.tif'))
    img_files = [ f[:-9] + '.tif' for f in mask_files]
    
    reshapedX = int(imgShapeX*reshapeFactor)
    reshapedY = int(imgShapeY*reshapeFactor)
    
    trainImgs = []
    trainSegs = []

    for (i, (impath, mpath)) in enumerate(zip(img_files, mask_files)):
        img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (reshapedX, reshapedY))
        trainImgs.append(img)
        
        img = cv.imread(mpath, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (reshapedX, reshapedY))
        trainSegs.append(img)        
        print (i, impath, mpath)
        
    return (trainImgs, trainSegs)
    
def rle_encode(img):
    prevVal = 0
    count = 0
    
    listOfRuns = []
    
    listPixels = img.reshape(img.shape[0]*img.shape[1], order='F')
    
    for i, val in enumerate(listPixels):
        if val!=0:
            count = count+1
        else:
            if prevVal!=0:
                listOfRuns.append((i-count+1, count))
                count = 0
        prevVal = val
    
    if count>0:            
        listOfRuns.append((len(listPixels)-count, count))

    rleString = ''    
    for px in listOfRuns:
        rleString += str(px[0]) + ' ' + str(px[1]) + ' '
    return rleString
    
                
        
    
    
if __name__ == "__main__":
    print('Hello')
    
    import time
    startTime = time.time()
    (trainImgs, trainSegs) = loadTrainData()
    loadTime = time.time() - startTime
    
    print 'Took', loadTime, 'secs to load the training data'

