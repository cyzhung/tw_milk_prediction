# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:25:37 2021

@author: 123
"""

import torch
import pandas as pd
import numpy as np
from model import Net

def test(model,test_data):
    check_point=torch.load('./model100.pkl')
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()
    result=[]
    with torch.no_grad():
        for data in ((test_data)):
            result.append(model(data).item())
            
    return result

            
if __name__=='__main__':
    test_data=pd.read_csv('./test.csv')

    ID=test_data['1']
    test_data.drop(['11','1'],axis=1,inplace=True)
    test_data.fillna(value=test_data.mean(),inplace=True)
    
    
    test_data=test_data.to_numpy()
    test_data=torch.from_numpy(test_data).float()
    num_input=test_data.shape[1]
    num_output=1

    model=Net(num_input,num_output)
    
    result=test(model,test_data)
    
    result=pd.DataFrame({'ID':ID,'1':result})
    result.set_index('ID',inplace=True)
    result.to_csv('submission.csv')
    
    