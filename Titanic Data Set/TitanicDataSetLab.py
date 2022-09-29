# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:29:50 2020

@author: 812976
"""

import numpy as np
import matplotlib.pyplot as plt

#normalizes x column using mean normalization
def normalize(xcol):
    return ( xcol - np.mean(xcol) ) / ( np.max(xcol) - np.min(xcol) )
    
#Creates TitanicSurvival Train Data
#Returns 2D x arr, and 2D y arr
def createTrainData():
    #Reads in file
    fileName = 'TitanicSurvival_Train.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    #Reads in data from file
    data = np.loadtxt(raw_data, usecols = (1, 2, 5, 6, 7, 8, 10), skiprows = 1, delimiter=",", dtype=np.str)

    #Adjusts sex column to represent data
    data[:,2][data[:,2]=="female"] = 1
    data[:,2][data[:,2]=="male"] = 0
    
    #if blank, replace with "not a number"
    data[:,3][data[:,3]==""] = np.nan
    
    #data is a matrix of Strings, convert before calculating mean
    age = data[:,3].astype(np.float)
    
    #get the mean of a row, ignoring nan, then replace nan with mean
    #numpy function which computes the mean, ignoring nan
    saved_mean = np.nanmean(age)
    age[np.isnan(age)] = saved_mean
    
    y = data[:,0].astype(np.float).reshape(len(data),1)
    pclass = data[:,1].astype(np.float).reshape(len(data),1)
    sex = data[:,2].astype(np.float).reshape(len(data),1)
    sibsp = data[:,4].astype(np.float).reshape(len(data),1)
    parch = data[:,5].astype(np.float).reshape(len(data),1)
    age = age.reshape(len(data),1)
    
    normalize(pclass)
    normalize(sex)
    normalize(age)
    normalize(sibsp)
    normalize(parch)
    
    bias = np.ones((len(age),1))
    x = np.concatenate((bias, pclass, sex, age, sibsp, parch), axis=1)
    return x, y

#Creates TitanicSurvival Test Data
#returns 2D x arr and 2D y arr
def createTestData():
    #Reads in file
    fileName = 'TitanicSurvival_Test.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')

    #Reads in data from file
    data = np.loadtxt(raw_data, usecols = (1, 4, 5, 6, 7, 9), skiprows = 1, delimiter=",", dtype=np.str)

    #Adjusts sex column to represent data
    data[:,1][data[:,1]=="female"] = 1
    data[:,1][data[:,1]=="male"] = 0
    
    #if blank, replace with "not a number"
    data[:,2][data[:,2]==""] = np.nan
    
    #data is a matrix of Strings, convert before calculating mean
    age = data[:,2].astype(np.float)
    
    #get the mean of a row, ignoring nan, then replace nan with mean
    #numpy function which computes the mean, ignoring nan
    saved_mean = np.nanmean(age)
    age[np.isnan(age)] = saved_mean
    
    pclass = data[:,0].astype(np.float).reshape(len(data),1)
    sex = data[:,1].astype(np.float).reshape(len(data),1)
    sibsp = data[:,3].astype(np.float).reshape(len(data),1)
    parch = data[:,4].astype(np.float).reshape(len(data),1)
    age = age.reshape(len(data),1)
    
    normalize(pclass)
    normalize(sex)
    normalize(age)
    normalize(sibsp)
    normalize(parch)
    
    bias = np.ones((len(age),1))
    x = np.concatenate((bias, pclass, sex, age, sibsp, parch), axis=1)
    return x

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
    
trainX, trainY = createTrainData()

weights = np.ones(6).reshape(6,1)
lr = 0.005
iterations = 0
maxIterations = 100000
vectDiff = 1
minDiff = 0.01
costArr = []

while iterations < maxIterations and vectDiff > minDiff:
    costArr = np.append(costArr, computeCost(trainX, weights, trainY))
    gradients = computeGradients(trainX, weights, trainY)
    weights -= lr*gradients
    iterations +=1
    vectDiff = np.linalg.norm(gradients)

#Uses the weights calculated from the
#train data set to give percents and predictions 
#For the test data set
testX = createTestData()
wx = np.dot(testX, weights)
predictions = sigmoid(wx)
percents = predictions*100
predictions[:,0][predictions[:,0] > 0.5]=1
predictions[:,0][predictions[:,0] < 0.5]=0

print("--------------------------------")
print("Train Data Set")
print("Final Cost: " + str(costArr[-1]))
print("Iterations: " + str(iterations))
print("Final Weights: " + str(weights))
print("I'm not that suprised about anything with my weights, at first I thought the age's weight was too small but then I realized that the age varis from like 5 to 70, so it actually makes alot of sense. I'd say all these weights make sense logically, with their signs too")
print("--------------------------------")
print("Test Data Set")
print("The Test Data Set doesn't have y values so I don't know how see how accurate the model is")
print("But after going through the model and looking at the weights for each factor, I'd say it appears pretty accurate.")
print("I know the model is accurate for the train data set since the cost is low, but unless I have the y, actual, values of the test data set I honestly don't know how to determine its accuracy with respect to the test data")
print("I tried googling the data set to see if I could only find it but after 15 minutes of looking I could find what appeared to be the same data set, but it had different information, like variables, or was limited to the first 839 passengers which makes up the train data set, not the test which is what I was looking for.")
print("In other words I gave up and decided to submit this as it is")
print("--------------------------------")

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(range(iterations),costArr)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost Function')
ax.set_title("Train Data - Cost")
fig.show()