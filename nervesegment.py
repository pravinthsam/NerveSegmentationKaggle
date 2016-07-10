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
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

imgShapeY = 420
imgShapeX = 580
reshapeFactor = 0.25

imgRows = int(imgShapeX*reshapeFactor)
imgCols = int(imgShapeY*reshapeFactor)


def loadTrainData():
    pathToData = './Data/train/'
    mask_files = sorted(glob.glob(pathToData + '*_mask.tif'))
    img_files = [ f[:-9] + '.tif' for f in mask_files]
    
    trainImgs = []
    trainSegs = []

    for (i, (impath, mpath)) in enumerate(zip(img_files, mask_files)):
        img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (imgRows, imgCols))
        trainImgs.append(img)
        
        img = cv.imread(mpath, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (imgRows, imgCols))
        trainSegs.append(img)        
        print (i, impath, mpath)
        if i == 499:
           break
        
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
    
def createKerasModel_isMask():
    model = Sequential()
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform',
                            input_shape=(1, imgCols, imgRows)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model
    
def createKerasModel_whereMask():
    model = Sequential()
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform',
                            input_shape=(1, imgCols, imgRows)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))

    # Final output layer    
    model.add(Convolution2D(1, imgRows, imgCols, border_mode='same', init='glorot_uniform'))
    model.add(Activation('sigmoid'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')
    
    return model
    
def createKerasModel_whereMask2():
    model = Sequential()
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform',
                            input_shape=(1, imgCols, imgRows)))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))
    
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.10))

    # Final output layer    
    model.add(Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform'))
    model.add(Activation('sigmoid'))
    
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    
    return model
    
def normaliseImageArray(imgs):
        
    for i, img in enumerate(imgs):
        imgs[i] = np.array(img, dtype='float32')/255.0
    
    return imgs    
        
    
                
        
    
    
if __name__ == "__main__":
    print('Hello')
    
    import time
    startTime = time.time()
    (trainImgs, trainSegs) = loadTrainData()
    loadTime = time.time() - startTime
    print 'Took', loadTime, 'secs to load the training data'
    '''
    isMaskPresent = isNotEmpty(trainSegs)
    
    print 'Mask is present in', float(np.sum(isMaskPresent)*100)/float(len(isMaskPresent)), '% of the training data'
    
    
    # Predict if mask is present
    model_isMask = createKerasModel_isMask()
    
#    from keras.utils.visualize_util import plot
#    plot(model1, to_file='model.png', show_shapes='True')
    trainImgs = normaliseImageArray(trainImgs)
    
    trainImgs = np.array(trainImgs)
    trainImgs = trainImgs.reshape(trainImgs.shape[0], 1, imgCols, imgRows)
    
    isMaskPresentCategorical = to_categorical(isMaskPresent, 2)
    
    model_isMask.fit(trainImgs, isMaskPresentCategorical, batch_size=100, nb_epoch=100, shuffle=True, verbose=1)
    
    trainPreds = model_isMask.predict(trainImgs, batch_size=100, verbose=1)
    
    isMaskPresentPred = [t[1]>t[0] for t in trainPreds]
    
    correctPreds = 0.0
    for i in range(len(isMaskPresent)):
        if isMaskPresent[i] == isMaskPresentPred[i]:
            correctPreds += 1.0
    
    print 'Training accuracy is', correctPreds/len(isMaskPresent), '%'
    '''
    model_mask = createKerasModel_whereMask2()
    from keras.utils.visualize_util import plot
    plot(model_mask, to_file='model.png', show_shapes='True')
    
    trainImgs = normaliseImageArray(trainImgs)
    trainImgs = np.array(trainImgs)
    #trainImgs = np.array(trainImgs, dtype='uint8')
    trainImgs = trainImgs.reshape(trainImgs.shape[0], 1, imgCols, imgRows)
    
    trainSegs = normaliseImageArray(trainSegs)
    trainSegs = np.array(trainSegs)
    trainSegs = trainSegs.reshape(trainSegs.shape[0], 1, imgCols, imgRows)
    
    model_mask.fit(trainImgs, trainSegs, batch_size=100, nb_epoch=10, shuffle=True, verbose=1)
    
    predMasks = model_mask.predict(trainImgs, batch_size=100, verbose=1)
    
    plt.imshow(predMasks[0][0], cmap='gray')  
    # plt.imshow(trainSegs[0][0], cmap='gray')  