# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:29:22 2020

@author: 812976
"""

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def createData():
    fileName = 'cancer_reg.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    data = np.loadtxt(raw_data, usecols = (2,4), skiprows = 1, delimiter=",")

    x = data[:,1:2]/10000
    y = data[:,0:1]

    bias = np.ones((len(x),1))
    x = np.concatenate((bias,x), axis=1)
    return x, y

x, y = createData()

w = np.array([0, 1.5]).reshape((2,1))
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(x[:,1:2],y[:,0:1],marker='x', color='Red')

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

lr = .01
iterations = 10000
costs = []
for i in range(iterations):
    w -= lr*computeGradient(w,x,y)
    costs.append(computeCost(w,x,y))
    
xvalues = x
yvalues = xvalues*w[1]+w[0]
ax.plot(xvalues, yvalues)
print(w)
fig.show()

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(iterations),costs)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost Function')
fig2.show()
