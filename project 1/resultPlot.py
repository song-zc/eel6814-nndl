# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:56:11 2020

@author: Jonathan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lossSum=np.zeros((1,100))
accSum=np.zeros((1,100))
trainLossSum=np.zeros((1,100))
trainAccSum=np.zeros((1,100))
data=pd.read_csv('data_nobn.csv',header=None)

for i in range (0,40,4):
    lossSum+=data.iloc[i,1:101].astype(float).to_numpy()
    accSum+=data.loc[i+1,1:101].astype(float).to_numpy()
    trainLossSum+=data.loc[i+2,1:101].astype(float).to_numpy()
    trainAccSum+=data.loc[i+3,1:101].astype(float).to_numpy() 


lossSum=(lossSum/10).reshape(100,1).tolist()
accSum=(accSum/10).reshape(100,1).tolist()
trainLossSum=(trainLossSum/10).reshape(100,1).tolist()
trainAccSum=(trainAccSum/10).reshape(100,1).tolist()


plt.plot (lossSum,label='Vali loss without BN')  
plt.plot (trainLossSum,label='Train loss without BN')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

#plt.subplot(122)
#plt.plot (accSum,label='Vali acc')  
#plt.plot (trainAccSum,label='Train acc')
#plt.xlabel('epoch')
#plt.ylabel('acc')
#plt.legend()

plt.show()

print (trainAccSum[99],accSum[99])