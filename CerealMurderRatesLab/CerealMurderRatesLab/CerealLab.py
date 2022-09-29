# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:51:42 2020

@author: 812976
"""

"""
Created on Thu Nov  5 16:51:42 2020

@author: 812976
"""

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def createData():
    fileName = 'Cereal.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    data = np.loadtxt(raw_data, usecols = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), skiprows = 1, delimiter=",")

    x1 = data[:,0:1]
    x2 = data[:,1:2]
    x3 = data[:,2:3]
    x4 = data[:,3:4]
    x5 = data[:,4:5]
    x6 = data[:,5:6]
    x7 = data[:,6:7]
    x8 = data[:,7:8]
    x9 = data[:,8:9]
    x10 = data[:,9:10]
    x11 = data[:,10:11]
    x12 = data[:,11:12]
    
    x1 = (x1 - np.mean(x1))/(np.max(x1)-np.min(x1))
    x2 = (x2 - np.mean(x2))/(np.max(x2)-np.min(x2))
    x3 = (x3 - np.mean(x3))/(np.max(x3)-np.min(x3))
    x4 = (x4 - np.mean(x4))/(np.max(x4)-np.min(x4))
    x5 = (x5 - np.mean(x5))/(np.max(x5)-np.min(x5))
    x6 = (x6 - np.mean(x6))/(np.max(x6)-np.min(x6))
    x7 = (x7 - np.mean(x7))/(np.max(x7)-np.min(x7))
    x8 = (x8 - np.mean(x8))/(np.max(x8)-np.min(x8))
    x9 = (x9 - np.mean(x9))/(np.max(x9)-np.min(x9))
    x10 = (x10 - np.mean(x10))/(np.max(x10)-np.min(x10))
    x11 = (x11 - np.mean(x11))/(np.max(x11)-np.min(x11))
    x12 = (x12 - np.mean(x12))/(np.max(x12)-np.min(x12))
    y = data[:,12:13]

    bias = np.ones((len(x1),1))
    x = np.concatenate((bias, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), axis=1)
    return x, y

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

x, y = createData()

w = np.ones(13).reshape(13,1)

lr = .05
maxI = 10000
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
print('y = ' + str(w[0]) + ' + ' + str(w[1]) + '*x1 + ' + str(w[2]) + '*x2')
print(' + ' + str(w[3]) + '*x3 + ' + str(w[4]) + '*x4 + ' + str(w[5]) + '*x5')
print(' + ' + str(w[6]) + '*x6 + ' + str(w[7]) + '*x7 + ' + str(w[8]) + '*x8')
print(' + ' + str(w[9]) + '*x9 + ' + str(w[10]) + '*x10 + ' + str(w[11]) + '*x11')
print(' + ' + str(w[12]) + '*x12')
print('----------------------------------------')

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(i),costs)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost Function')
fig2.show()