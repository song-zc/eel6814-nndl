# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:29:53 2020

@author: Jonathan
"""
import os
import time

timeStart=time.time()
for i in range(0,10):
#    exec(open("train.py").read())
    runfile('C:/Users/Jonathan/OneDrive - University of Florida/nn/project 1/mlpTrain.py', wdir='C:/Users/Jonathan/OneDrive - University of Florida/nn/project 1')

timeEnd=time.time()
print ('time',timeEnd-timeStart)