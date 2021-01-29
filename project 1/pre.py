# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:01:50 2020

@author: Jonathan
"""
import sys
sys.path.append("fashion-mnist-master/")
from utils import mnist_reader
import numpy as np
import pandas as pd

X_train, y_train = mnist_reader.load_mnist('fashion-mnist-master/data/fashion/', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion-mnist-master/data/fashion/', kind='t10k')

testData=pd.DataFrame(data=X_test)
testData['class']=y_test
rawData=pd.DataFrame(data=X_train)
rawData['class']=y_train
trainData=pd.DataFrame()
valiData=pd.DataFrame()
for i in range (0,10):
    temp=rawData[rawData['class']==i]
    train,vali=np.split(temp.sample(frac=1),[int(len(temp)/3*2)])
    trainData=pd.concat([trainData,train])
    valiData=pd.concat([valiData,vali])

trainData=trainData.sample(frac=1).reset_index(drop=True)
valiData=valiData.sample(frac=1).reset_index(drop=True)     

#trainData.to_csv('trainData.csv', index=False)
#valiData.to_csv('valiData.csv', index=False)
#testData.to_csv('testData.csv', index=False)
#    