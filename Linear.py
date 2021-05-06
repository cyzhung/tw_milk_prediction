# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:44:21 2021

@author: 123
"""
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def load_data():
    train_data=pd.read_csv("./train.csv")
    label=(train_data[["11"]])
    label=label.to_numpy()

    train_data.drop(['1','11'],inplace=True,axis=1)
    train_data=train_data.to_numpy()

    return train_data,label
    

if __name__=='__main__':
    train_data,label=load_data()
    X_train,X_test, y_train, y_test =train_test_split(train_data,label,test_size=0.2)
    print(y_train.shape)
    print(y_test.shape)
    model=LinearRegression()
    model.fit(X_train,y_train)
    test=pd.read_csv('./test.csv')
    data=test.drop(['1','11'],axis=1)
    ID=test['1']
    ans=model.predict(data)
    ans=ans.reshape(ans.shape[0])
    sub=pd.DataFrame({'ID':ID,'1':ans})
    sub.set_index('ID',inplace=True)
    sub.to_csv('submissionLinear.csv')
    print("決定係數(train):{:.3f}".format(model.score(X_train,y_train)))
    print("決定係數(test):{:.3f}".format(model.score(X_test,y_test)))