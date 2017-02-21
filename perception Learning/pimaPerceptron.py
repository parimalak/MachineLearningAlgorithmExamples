
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Demonstration of the Perceptron on the Pima Indian dataset

# Modified to add linear regression and 2-fold cross-validation

import numpy as np
import pcn
import linreg

pima = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')

# Various preprocessing steps
pima[np.where(pima[:,0]>8),0] = 8
pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60),7] = 5

pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)

inputs1 = pima[::2,:8]
inputs2 = pima[1::2,:8]
targets1 = pima[::2,8:9]
targets2 = pima[1::2,8:9]

# Perceptron training on the preprocessed dataset
p1 = pcn.pcn(inputs1,targets1)
p1.pcntrain(inputs1,targets1,0.25,100)
cm1 = p1.confmat(inputs2,targets2)
p2 = pcn.pcn(inputs2,targets2)
p2.pcntrain(inputs2,targets2,0.25,100)
cm2 = p2.confmat(inputs1,targets1)
cm = cm1 + cm2
print("Perceptron classification accuracy: ")
print(np.trace(cm)/np.sum(cm))

# Linear regression on the preprocessed dataset
beta1 = linreg.linreg(inputs1,targets1)
beta2 = linreg.linreg(inputs2,targets2)
inputs1 = np.concatenate((inputs1,-np.ones((np.shape(inputs1)[0],1))),axis=1)
inputs2 = np.concatenate((inputs2,-np.ones((np.shape(inputs2)[0],1))),axis=1)
outputs1 = np.dot(inputs1,beta2)
outputs1[np.where(outputs1>0.5)] = 1
outputs1[np.where(outputs1<=0.5)] = 0
outputs2 = np.dot(inputs2,beta1)
outputs2[np.where(outputs2>0.5)] = 1
outputs2[np.where(outputs2<=0.5)] = 0
outputs = np.r_[outputs1,outputs2]
targets = np.r_[targets1,targets2]
print("Linear regression classification accuracy:")
print(np.sum(targets==outputs)/np.size(targets))
