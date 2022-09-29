# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:51:42 2020

@author: 812976
"""

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def createData():
    fileName = 'murdersunemployment.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    data = np.loadtxt(raw_data, usecols = (2,3,4), skiprows = 1, delimiter=",")

    x1 = data[:,0:1]
    x2 = data[:,1:2]
    y = data[:,2:3]

    return x1, x2, y

def computeGradient(w, X, Y):
    pre = np.dot(X,w)
    diff = pre-Y
    grad = np.dot(X.T,diff)
    return grad/len(X)

def computeCost(w, X, Y):
    s = np.dot(X, w)-Y
    s = s**2
    sum_error_squared = np.sum(s)
    return sum_error_squared/len(X)

x1, x2, y = createData()

x1 = (x1 - np.mean(x1))/(np.max(x1)-np.min(x1))
x2 = (x2 - np.mean(x2))/(np.max(x2)-np.min(x2))

bias = np.ones((len(x1),1))
x = np.concatenate((bias, x1, x2), axis=1)
w = np.array([[0],
              [1.5],
              [1.5]])
lr = .05
maxI = 1000
costs = []
i = 0
vectDiff = np.linalg.norm(computeGradient(w,x,y))
minDiff = 0.1

while (i < maxI) and (vectDiff > minDiff):
   gradients = computeGradient(w,x,y)
   w -= lr*gradients
   costs.append(computeCost(w,x,y))
   vectDiff = np.linalg.norm(gradients)
   i+=1
   
print('----------------------------------------')
print('Iterations: ' + str(i))
print('Max Iterations: ' + str(maxI))
print('Learning Rate: ' + str(lr))
print('Weights: ' + str(w))
print('Cost: ' + str(costs[len(costs)-1]))
print('----------------------------------------')
print('y = murders per annum per 1,000,000 ')
print('x1 = percent with incomes below $5,000')
print('x2 = percent unemployed')
print('Model:')
print('y = ' + str(w[0]) + ' + ' + str(w[1]) + '*x1 ' + str(w[2]) + '*x2')
print('----------------------------------------')

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(i),costs)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost Function')
fig2.show()
