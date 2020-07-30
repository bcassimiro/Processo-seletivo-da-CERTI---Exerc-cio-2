# -*- coding: utf-8 -*-

# SUPPORT VECTOR MACHINE WITH HOG
#------------------------------------------------------------------------------
# Objective: To train a SVM with HOG to identify 'thumbs up' and 'thumbs down'.
#------------------------------------------------------------------------------
# Status (last update in 29/07/2020): 
# - Ok!
#------------------------------------------------------------------------------

# Importing libraries ---------------------------------------------------------
import cv2
import numpy as np
import os
from sklearn import svm 
from skimage.feature import hog
from skimage import exposure

# Reading the images (training) -----------------------------------------------
dirTRAINING = 'training'
listTRAINING = os.listdir(dirTRAINING)
numTRAINING = len(listTRAINING)

training = np.zeros((numTRAINING,24*32)).astype(np.float64)
for img in range(1,numTRAINING):
    imageNAME = ''.join([dirTRAINING,'\\',listTRAINING[img]])
    image = cv2.imread(imageNAME,0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4,4),
                    cells_per_block=(1,1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1,24*32).astype(np.float64)
    training[img,:] = imageLINE

# cv2.imshow('Image',image)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Reading the images (test) ---------------------------------------------------
dirTEST = 'test'
listTEST = os.listdir(dirTEST)
numTEST = len(listTEST)
    
test = np.zeros((numTEST,24*32)).astype(np.float64)
for img in range(1,numTEST):
    imageNAME = ''.join([dirTEST,'\\',listTEST[img]])
    image = cv2.imread(imageNAME,0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4,4),
                    cells_per_block=(1,1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1,24*32).astype(np.float64)
    test[img,:] = imageLINE

# cv2.imshow('Image',image)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Reading the images (generalization) -----------------------------------------
dirGENERALIZATION = 'generalization'
listGENERALIZATION = os.listdir(dirGENERALIZATION)
numGENERALIZATION = len(listGENERALIZATION)

generalization = np.zeros((numGENERALIZATION,24*32)).astype(np.float64)
for img in range(1,numGENERALIZATION):
    imageNAME = ''.join([dirGENERALIZATION,'\\',listGENERALIZATION[img]])
    image = cv2.imread(imageNAME,0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4,4),
                    cells_per_block=(1,1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255)) 
    imageLINE = image.reshape(-1,24*32).astype(np.float64)
    generalization[img,:] = imageLINE

# cv2.imshow('Image',image)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Reading the labels ----------------------------------------------------------
# trainingCSV = 'trainingLABELS.csv'
# trainingLABELS = np.genfromtxt(trainingCSV, delimiter = '')
trainingLABELS = np.concatenate((np.ones(int(numTRAINING/2)),np.zeros(int(numTRAINING/2))))

# testCSV = 'testLABELS.csv'
# testLABELS = np.genfromtxt(testCSV, delimiter = '')
testLABELS = np.concatenate((np.ones(int(numTEST/2)),np.zeros(int(numTEST/2))))

# generalizationCSV = 'generalizationLABELS.csv'
# generalizationLABELS = np.genfromtxt(generalizationCSV, delimiter = '')
generalizationLABELS = np.array([0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1.,
       1., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0.,
       1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 1.])


# Training the SVM ------------------------------------------------------------
clf = svm.SVC()
clf.fit(training, trainingLABELS)


# Veryfing the accuracy of the test -------------------------------------------
result = clf.predict(test)

matches = result == testLABELS
correct = np.count_nonzero(matches)
accuracy = correct*100.0/len(result)

# Veryfing the accuracy of the generalization ---------------------------------
resultGEN = clf.predict(generalization)

matchesGEN = resultGEN == generalizationLABELS
correctGEN = np.count_nonzero(matchesGEN)
accuracyGEN = correctGEN*100.0/len(resultGEN)