# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:29:50 2020

@author: 812976
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

#Creates TitanicSurvival Train Data
#Returns 2D x arr, and 2D y arr
def createTrainData():
    #Reads in file
    fileName = 'irisdata.csv'
    print("fileName: ", fileName)
    print("Wait like a minute")
    raw_data = open(fileName, 'rt')

    #Reads in data from file
    data = np.loadtxt(raw_data, usecols = (0, 1, 2, 3, 4), skiprows = 1, delimiter=",", dtype=np.float)   
    
    y = data[:,4:]
    forlater = data[:,4:]
    ohe = OneHotEncoder(categories='auto')
    y = ohe.fit_transform(y).toarray()
    
    x = data[:, 0:4]
    bias = np.ones((len(data),1))
    x = np.concatenate((bias, x), axis=1)
    return x, y, forlater

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, w, y):
    z = np.dot(X, w)
    temp = sigmoid(z)
    costs = y*np.log(temp) + (1-y)*np.log(1-temp)
    return -np.mean(costs)

def computeGradients(X, w, Y):
    z = np.dot(X, w)
    temp = sigmoid(z)
    grads = (temp - Y)
    grads = np.dot(np.transpose(X), grads)
    return grads/len(X)

lr = 0.005
maxIterations = 200000
vectDiff = 1
minDiff = 0.005

#Normalizes the data sets
trainX, trainY, fornow = createTrainData()

y1 = trainY[:,0]
y2 = trainY[:,1]
y3 = trainY[:,2]

meanX = np.mean(trainX[:, 1:5], axis=0)
maxX = np.max(trainX[:, 1:5], axis=0)
minX = np.min(trainX[:, 1:5], axis = 0)
trainX[:,1:5] = (trainX[:, 1:5] - meanX) / (maxX - minX)

weights1 = np.ones(5)
iterations1 = 0
costArr1 = []
while iterations1 < maxIterations and vectDiff > minDiff:
    costArr1 = np.append(costArr1, computeCost(trainX, weights1, y1))
    gradients = computeGradients(trainX, weights1, y1)
    weights1 -= lr*gradients
    iterations1 +=1
    vectDiff = np.linalg.norm(gradients)
    
vectDiff = 1
weights2 = np.ones(5)
iterations2 = 0
costArr2 = []
while iterations2 < maxIterations and vectDiff > minDiff:
    costArr2 = np.append(costArr2, computeCost(trainX, weights2, y2))
    gradients = computeGradients(trainX, weights2, y2)
    weights2 -= lr*gradients
    iterations2 +=1
    vectDiff = np.linalg.norm(gradients)
    
vectDiff = 1
weights3 = np.ones(5)
iterations3 = 0
costArr3 = []
while iterations3 < maxIterations and vectDiff > minDiff:
    costArr3 = np.append(costArr3, computeCost(trainX, weights3, y3))
    gradients = computeGradients(trainX, weights3, y3)
    weights3 -= lr*gradients
    iterations3 +=1
    vectDiff = np.linalg.norm(gradients)
    
print("Iris Data Set")
print("-----------------------------------")
print("Setatosa")
print("Final Cost: " + str(costArr1[-1]))
print("Iterations: " + str(iterations1))
print("Final Weights: " + str(weights1))
print("-----------------------------------")
print("Versicolor")
print("Final Cost: " + str(costArr2[-1]))
print("Iterations: " + str(iterations2))
print("Final Weights: " + str(weights2))
print("-----------------------------------")
print("Virginia")
print("Final Cost: " + str(costArr3[-1]))
print("Iterations: " + str(iterations3))
print("Final Weights: " + str(weights3))
print("-----------------------------------")

predictions1 = sigmoid(np.dot(trainX, weights1)).reshape(len(trainX),1)
predictions2 = sigmoid(np.dot(trainX, weights2)).reshape(len(trainX),1)
predictions3 = sigmoid(np.dot(trainX, weights3)).reshape(len(trainX),1)
predictions = np.concatenate((predictions1, predictions2, predictions3), axis=1)

outcomes = []
errIndices = []
numWrong = 0
for i in range(len(predictions)):
    outcomes = np.append(outcomes, np.argmax(predictions[i][:]) + 1)
    if (np.argmax(predictions[i][:]) + 1) != fornow[i][:]:
        numWrong+=1
        errIndices = np.append(errIndices, i)
        
print("Results:")
print("Number wrong: " + str(numWrong))
print("Indices of Wrong: " + str(errIndices))
print("-----------------------------------")