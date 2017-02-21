# -*- coding: utf-8 -*-
"""
Homework 1 Problem 5: Gaussian-distributed naive Bayes.

#Demonstartion of Naive Bayes real-valued numeric data 
generated using gaussian mixture model
"""

import numpy as np
import scipy.stats as st
from gmm import twoDimGmm

def naiveBayes(trainData,trainLabel,testData):
    n,p = np.shape(trainData)
    k = np.int_(np.max(trainLabel)) + 1
    N = np.shape(testData)[0]
    
    # Estimate mean and standard deviation along each dimension for each class
    meanArray = np.zeros((k,p))
    sigmaArray = np.zeros((k,p))
    for c in range(k):
        inClass = np.where(trainLabel == c)
        meanArray[c,:] = np.mean(trainData[inClass],0)
        sigmaArray[c,:] = np.std(trainData[inClass],0)
    
    # Evaluate Gaussian with estimated mean and standard deviation for each
    # class in each dimension for each test data point
    testProb = np.ones((N,k))
    for c in range(k):
        for dim in range(p):
            testProb[:,c] *= st.norm.pdf(testData[:,dim],meanArray[c,dim],
                                    sigmaArray[c,dim])
    
    # Set predicted class of each data point to class with highest estimated
    # probability
    predClass = np.argmax(testProb,1)
    
    return predClass,testProb

# Generate data from 2-D GMM and split into training and test
n = 100
coords,labels = twoDimGmm(n,-2,2)
trainData = coords[:80,:]
trainLabel = labels[:80]
testData = coords[80:,:]
testLabel = labels[80:]

# Apply naive Bayes classifier and return both predicted class and
# posterior probability for each class for each data point
predClass,testProb = naiveBayes(trainData,trainLabel,testData)

accuracy = np.size(np.where(testLabel == predClass)) / np.size(testLabel)
print("Naive Bayes classifier accuracy: " + "%.2f" % accuracy)
