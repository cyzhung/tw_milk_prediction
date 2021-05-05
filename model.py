# -*- coding: utf-8 -*-
"""
Created on Sat May  1 17:12:20 2021

@author: 123
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Net(nn.Module):
    def __init__(self,num_input,num_output):
        super(Net,self).__init__()
        self.fc1=nn.Linear(num_input,512)

        self.fc2=nn.Linear(512,256)
        
        self.fc3=nn.Linear(256,128)
        self.fc4=nn.Linear(128,64)
        self.fc5=nn.Linear(64,32)
        self.fc6=nn.Linear(32,1)
        
        self.drop=nn.Dropout(0.1)
    def forward(self,x):

        x=F.leaky_relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))

        x=F.relu(self.fc4(x))

        x=F.relu(self.fc5(x))
       # x=self.drop(x)
        x=self.fc6(x)
        return x
        
device='cuda'
if __name__=='__main__':
    net=Net(7,1).to(device)
    data=np.random.rand(7)
    
    data=torch.tensor(data).to(device,dtype=torch.float)
    predict=net(data)
    print(predict)