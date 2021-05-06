# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:35:21 2021

@author: 123
"""
#%%
import pandas as pd
import numpy as np
import time

SECOND_TO_MONTH=2592000
def preprocess():
    birth=pd.read_csv('./data/birth.csv')
    breed=pd.read_csv('./data/breed.csv')
    report=pd.read_csv('./data/report.csv')
    spec=pd.read_csv('./data/spec.csv')
    submission=pd.read_csv('./data/submission.csv')
    
    report.set_index('1',inplace=True)

        
        
                                          
    report=report[['2','3','4','5','8','9','10','11','12','14','19','20']]      #2:資料年度/3:資料月份/4:農場代號/5:乳牛編號/6:父親編號//7:母親編號8:出生日期/9:胎次/10:泌乳天數
                                                                        #11:乳量/12:這次生產/14:月齡/19:上次生產/20:第一次配種日期
    birth=birth[['1','6']]                                              #1:乳牛編號/6:乳牛體重

    spec=spec[['1','2','4','7']]
    

    #處理report  計算第一次生產時幾歲
    report["22"]=np.zeros(len(report))              #'22'存放第一胎為幾歲
    report["23"]=np.zeros(len(report))              #'23'存放產犢間隔
    report["24"]=np.zeros(len(report))              #'24'存放體重
    report['24'].replace(0,np.nan,inplace=True)
    report['25']=np.zeros(len(report))              #'25'存放健康狀態
    report['26']=np.zeros(len(report))              #'26'存放當時的季節
    d={}                                            #dict 用來看此乳牛是否已經計算過第一次生產日期





    #新增健康狀態
    for i in range(len(spec)):
        number=spec.loc[i,['1']]
        date=spec.loc[i,'4']
        dd=time.strptime(date,"%Y/%m/%d %H:%M")
        idx=report['5']==int(number)
        s_farm=spec.loc[i,'7']

        if(len(report.loc[idx,['2','3']])>0):

            for j in report.index[report['5']==int(number)]:

                r_farm=report.loc[j,'4']
                r_year=report.loc[j,'2']
                r_month=report.loc[j,'3']

                if(r_year==dd.tm_year and r_month == dd.tm_mon and r_farm==s_farm):
                    report.loc[j,'25']=spec.loc[i,'2']


    

                



    #將ABC改為牧場所在地
    report['4'].replace('A',"桃園",inplace=True)
    report['4'].replace('B',"彰化",inplace=True)
    report['4'].replace('C',"屏東",inplace=True)
    

    #將健康狀態的小寫n轉成大寫
    spec['2'].replace('n','N',inplace=True)

        
    #新增體重
    for i in range(len(birth)):
        number=birth.loc[i,['1']]
        weight=birth.loc[i,['6']].item()
        idx=report['5']==int(number)
        report.loc[idx,['24']]=weight

    

    #計算第一次生產為幾歲  計算產犢間隔  計算當時的季節
    for i in range(1,len(report)+1):
        month=report.loc[i,['3']].item()
        if(month>=3 and month <=5):
            report.loc[i,'26']='spring'
        elif (month>=6 and month <=8):
            report.loc[i,'26']='summer'
        elif (month>=9 and month <=11):
            report.loc[i,['26']]='fall'
        else:
            report.loc[i,['26']]='winter'
        
        #計算上次產犢及這次的間隔
        last_birth=report.loc[i,'19']
        this_birth=report.loc[i,'12']
        if(pd.isnull(last_birth)):
            report.loc[i,'23']=0
        else:
            last_birth=time.strptime(last_birth,"%Y/%m/%d %H:%M")       #轉為time模組格式
            this_birth=time.strptime(this_birth,"%Y/%m/%d %H:%M")       #轉為time模組格式
            
            last_birth=time.mktime(last_birth)                          #將時間轉為從1900/1/1到出生時間所經過的秒數
            this_birth=time.mktime(this_birth)
                
            month=(this_birth-last_birth)/SECOND_TO_MONTH/12            #將相差的秒數轉為月份 即可算出第一次生產歲數
            report.loc[i,['23']]=(month)                           

    
        
        #計算第一次生產
        number=str(report.loc[i,['5']].item())                 #母牛編號
        birth_time=report.loc[i,['8']].item()                  #乳牛出生日期
        child_birth=report.loc[i,['20']].item()                #乳牛第一胎日期
        if number in d:
            report.loc[i,['22']]=d[number]        
            continue


        if(pd.isnull(child_birth)):
            report.loc[i,['22']]=np.nan
        else:
            birth_time=time.strptime(birth_time,"%Y/%m/%d %H:%M")       #將出生時間轉為time模組格式
            child_birth=time.strptime(child_birth,"%Y/%m/%d %H:%M")     #將第一胎日期轉為time模組格式

            birth_time=time.mktime(birth_time)                          #將出生時間轉為從1900/1/1到出生時間所經過的秒數
            child_birth=time.mktime(child_birth)                        #將第一胎時間轉為從1900/1/1到出生時間所經過的秒數     
            
            month=(child_birth-birth_time)/SECOND_TO_MONTH              #將相差的秒數轉為月份 即可算出第一次生產歲數
            report.loc[i,['22']]=int(month)                            
            d[number]=int(month)
            
    


    #將所屬牧場轉成one-hot vector
    onehot=pd.get_dummies(report['4'])
    report=report.join(onehot)

    #將乳牛編號轉成one-hot vector
    onehot=pd.get_dummies(report['5'])
    report=report.join(onehot)

    #將季節轉為one-hot vector
    onehot=pd.get_dummies(report['26'])
    report=report.join(onehot)

    #將nan填入值
    report['22'].fillna(value=report['22'].mean(),inplace=True)
    report['24'].fillna(value=birth['6'].mean(),inplace=True)
    report['10'].fillna(value=report['10'].mean(),inplace=True)

    print(report.head())
    #提取測試資料
    ID=[]
    for i in range(len(submission)):
        ID.append(submission.loc[i,['ID']].item())
        
    test_data=report.drop(['2','3','4','5','8','12','19','20','26'],axis=1) #將不需要用來訓練的資料丟掉
    test_data=test_data.loc[ID]
    report.drop(index=ID,inplace=True)
    

    
    #寫檔
    test_data.to_csv('test.csv')

    train=report.drop(['2','3','4','5','8','12','19','20','26'],axis=1)  #將不需要用來訓練的資料丟掉
    train.to_csv('train.csv')


if __name__=='__main__':
   preprocess()

   
   
   
   
   
   
   
   
   
   
   
   
   


# %%

# %%
""