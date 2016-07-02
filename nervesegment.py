# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:30:26 2016

@author: pravinth
"""

import glob
import cv2 as cv
import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.optimizers import SGD

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
    
def rleEncode(img):
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
    
def isNotEmpty(listOfImgs):
    return [np.sum(img) > 10 for img in listOfImgs]
    
def createKerasModel1():
    model = Sequential()
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform', input_shape=(1, 105, 145)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform', input_shape=(1, 105, 145)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model  
    
                
        
    
    
if __name__ == "__main__":
    print('Hello')
    
#    import time
#    startTime = time.time()
#    (trainImgs, trainSegs) = loadTrainData()
#    loadTime = time.time() - startTime
#    
#    print 'Took', loadTime, 'secs to load the training data'
    
#    isMaskPresent = isNotEmpty(trainSegs)
#    
#    print 'Mask is present in', float(np.sum(isMaskPresent)*100)/float(len(isMaskPresent)), '% of the training data'
    
    # Predict if mask is present
    model1 = createKerasModel1()
    from keras.utils.visualize_util import plot
    plot(model1, to_file='model.png')
    
    

