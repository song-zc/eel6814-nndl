# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:11:23 2020

@author: Jonathan
"""
import numpy as np
import torch
import pandas as pd
from torch import nn, optim
#import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
#from torch.utils.data.dataset import Dataset
#from torchvision import transforms
import matplotlib.pyplot as plt

batchSize=100   
epochNum=100
learningRate = 0.001    

file= open("result.txt", "a+")
#class MyDataSet(Dataset):
#    def __init__(self, data,labels,transform=None,targetTransform=None):
#        self.data = data
#        self.labels=labels
#        self.transform = transform
# 
#    def __len__(self):
#        return len(self.data)
# 
#    def __getitem__(self,index):
#        if torch.is_tensor(index):
#            index = index.tolist()
#        data=self.data[index]
#        labels=self.labels[index]
#        if self.transform is not None:
#            data = self.transform(data)
#        if self.targetTransform is not None:
#            labels = self.transform(labels)
#        return data, labels


trainData=pd.read_csv('trainData.csv')
target = trainData['class']
del trainData['class']
trainSet = TensorDataset(torch.Tensor(np.array(trainData)), torch.Tensor(np.array(target)))

valiData=pd.read_csv('valiData.csv')
target = valiData['class']
del valiData['class']
valiSet = TensorDataset(torch.Tensor(np.array(valiData)), torch.Tensor(np.array(target)))
#trainSet=MyDataSet(trainData.drop('class', axis=1),trainData['class'])
#valiSet=MyDataSet(valiData.drop('class', axis=1),valiData['class'])

trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
valiLoader = DataLoader(valiSet, batch_size=batchSize, shuffle=False)



#class Cnn(nn.Module):
#    def __init__(self,inDim,outDim):
#        super(Cnn,self).__init__()
#        self.conv1 = nn.Conv2d(inDim, 6, 5, 1, 2)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
#        self.fc1 = nn.Linear(400, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, outDim)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
class Cnn(nn.Module):
    def __init__(self, inDim, outDim):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inDim, 24, 3, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.6)
            )

        self.fc = nn.Sequential(
            nn.Linear(1728,outDim))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

model=Cnn(1,10)
model=model.to(DEVICE)

optimizer = optim.SGD(params=model.parameters(),lr=learningRate)
#optimizer = optim.Adam(params=model.parameters(),betas=(0.9, 0.999), lr=learningRate)
criterion = nn.CrossEntropyLoss()

lossMin=1
trainLossPlot=[]
trainAccPlot=[]
lossPlot=[]
accPlot=[]

for epoch in range(epochNum):
    model.train()
    runningLoss = 0.0
    runningAcc=0.0
    print('epoch {}'.format(epoch + 1))
    print('=' * 10)
    
    for data in trainLoader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels=labels.long()
        inputs=inputs.view(batchSize,1,28,28)
        inputs=inputs.to(DEVICE)
        labels=labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        runningLoss += loss.item()
        
        _, pred = torch.max(outputs, 1)
        numCorrect = (pred == labels).sum()
        runningAcc += numCorrect.item()
        
    runningLoss=runningLoss/len(trainLoader)
    runningAcc=runningAcc/len(trainData)
    trainLossPlot.append(runningLoss)
    trainAccPlot.append(runningAcc)
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, runningLoss,runningAcc))
    
    model.eval()
    evalLoss = 0
    evalAcc = 0
    for data in valiLoader:
        inputs, labels = data
        labels=labels.long()
        inputs=inputs.view(batchSize,1,28,28)
        inputs=inputs.to(DEVICE)
        labels=labels.to(DEVICE)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        evalLoss += loss.item()
        _, pred = torch.max(outputs, 1)
        numCorrect = (pred == labels).sum()
        evalAcc += numCorrect.item()
    
    evalAcc=evalAcc/len(valiData)
    evalLoss=evalLoss/len(valiLoader)
    lossPlot.append(evalLoss)
    accPlot.append(evalAcc)
    print('Eval Loss: {:.6f}, Acc: {:.6f}'.format(evalLoss,evalAcc))
    if evalLoss<=lossMin:
        lossMin=evalLoss
        lossMinEpoch=epoch
    elif epoch-lossMinEpoch>20:
        break;
#    if evalLoss<=lossMin:
#        lossMin=evalLoss
#    else:
#        break
#torch.save(model,'model_24_48_0.6_2')

#outputData=pd.DataFrame(index=['lossPlot','accPlot','trainLossPlot','trainAccPlot'],data=[lossPlot,accPlot,trainLossPlot,trainAccPlot])
#outputData.to_csv('data_16_32_100_std_drop0.8.csv', mode='a', header=False)         
print (lossMin)
plt.plot (np.asarray(lossPlot),label='Vali loss SGD')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.plot (np.asarray(trainLossPlot),label='Train loss SGD')
plt.legend()
print ('Batch size=',batchSize)
print ('lr=',learningRate) 
file.close()