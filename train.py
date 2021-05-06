# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:13:58 2021

@author: 123
"""
#4:農場代號/5:乳牛編號/8:出生日期/9:胎次/10:泌乳天數
#11:乳量/12:這次生產/14:月齡/19:上次生產/20:第一次配種日期
import torch
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from model import Net
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl

batch_size=32
device='cuda'

class dataset(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx],self.label[idx]
    
    


def load_data():
    train_data=pd.read_csv("./train.csv")
    label=(train_data[["11"]])
    label=label.to_numpy()

    train_data.drop(['1','11'],inplace=True,axis=1)
    train_data=train_data.to_numpy()

    train_data=torch.from_numpy(train_data)

    
    



    
    return train_data,label

def train(model,trainloader,testloader):
    e=50
    for episode in range(1,e+1):
        
        train_loss=0.0
        
        for i,(X,y) in enumerate(trainloader):
            X=X.to(device)
            y=y.to(device)

            predict=model(X)
            loss=torch.sqrt(F.mse_loss(predict,y))
            
            opt.zero_grad()
            loss.backward()
            model.double()
            opt.step()
            train_loss+=loss.item()
        
            

        train_loss/=i
        
        test_loss=0.0
        for i,(X,y) in enumerate(testloader):
            X=X.to(device)
            y=y.to(device)
            
            predict=model(X)
            loss=torch.sqrt(F.mse_loss(predict,y))
            

            test_loss+=loss.item()
            
        test_loss/=i
        
        
        
        print("Episode:{},Train_Loss:{},Test_loss:{}".format(episode,np.round(train_loss,2),np.round(test_loss,2)))
        
    
        if episode % 10 ==0:
            torch.save({
                    'epoch':e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                    }, './model{}.pkl'.format(episode))
            
    
            
            
if __name__=='__main__':
    train_data,label=load_data()
    
    X_train,X_test, y_train, y_test =train_test_split(train_data,label,test_size=0.2)


    
    trainset=dataset(X_train,y_train)
    testset=dataset(X_test,y_test)
    

    trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testloader=DataLoader(testset,batch_size=batch_size,shuffle=True)
    
    num_input=train_data.shape[1]
    num_output=1
    model=Net(num_input,num_output).to(device,dtype=torch.double)
    
    opt=optim.Adam(model.parameters(),lr=0.001)
    
    
    train(model,trainloader,testloader)
    
    #test(model,X_train,X_test,y_train,y_test)
    

    
    