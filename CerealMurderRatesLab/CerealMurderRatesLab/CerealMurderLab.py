# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:51:42 2020

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