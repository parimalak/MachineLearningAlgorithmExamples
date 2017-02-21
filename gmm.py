# -*- coding: utf-8 -*-
"""
@Author :parimala killada
#HW1
#simulated data set of 100 data points 
in 2 dimensions with 2 classes using the following process for each data point:
1. Generated the class label of the data point 
#by first sampling a uniform random variable between 0 and 1. 
If the value is above 0.5, set the class to 0; otherwise, set the class to 1.
2. Generated each coordinate (dimension) of the data point 
by sampling a Gaussian random variable with class-dependent mean and standard deviation 
1. If the data point is in class 0, set the mean to -2; 
if it is in class 1, set the mean to 2.
This is a special case of a general model used in machine learning called 
the Gaussian mixture model.
Plotted the data points in 2 dimensions using red markers for class 0 and 
blue markers for class 

"""

import numpy as np
import matplotlib.pyplot as pl

def twoDimGmm(n,mean0,mean1):
    # Generate class labels by sampling from a uniform(0,1) distribution.
    # Set all values > 0.5 to 1 and all others to 0
    labels = np.random.rand(n)
    labels[np.where(labels>0.5)] = 0
    labels[np.where(labels>0)] = 1
    
    # Generate all data points from Gaussian distribution with mean 0 and
    # standard deviation 1, then add -2 to points in class 0 and add 2 to
    # points in class 1.
    coords = np.random.randn(n,2)
    class0 = np.where(labels==0)
    class1 = np.where(labels==1)
    coords[class0,:] = coords[class0,:] + mean0
    coords[class1,:] = coords[class1,:] + mean1
    
    return coords,labels

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    n = 100
    mean0 = -2
    mean1 = 2
    coords,labels = twoDimGmm(n,mean0,mean1)
    
    # Plot data points using red markers for class 0 and blue for class 1
    class0 = np.where(labels==0)
    class1 = np.where(labels==1)
    pl.ion()
    pl.figure()
    pl.plot(coords[class0,0],coords[class0,1],'ro')
    pl.plot(coords[class1,0],coords[class1,1],'bx')
    pl.show()
