# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:28:36 2021

@author: Junkai
"""
from datetime import datetime
import numpy as np
import pandas as pd
from os import listdir
import os
from os.path import isfile, join

mypath = os.getcwd()
city = 'Denver'
mypath1 = mypath + '\\Weather\\' + city + '\\Denver.csv'
mypath= mypath + '\\Energy\\' + city

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
demands = []

for f in onlyfiles:
    data = pd.read_csv(mypath+'\\'+f)
    ind = data.index[data['Date & Time'] == '2015-12-31 21:00'].tolist() + \
        data.index[data['Date & Time'] == '2015-01-01 05:00'].tolist()
    if not ind:
        ind = data.index[data['Date & Time'] == '2015-12-31 21:00:00'].tolist() + \
            data.index[data['Date & Time'] == '2015-01-01 05:00:00'].tolist()
    if not ind:
        ind = data.index[data['Date & Time'] == '12/31/2015 21:00'].tolist() + \
            data.index[data['Date & Time'] == '1/1/2015 5:00'].tolist()            
    data = data[ind[0]:ind[1]+1]
    data = data.iloc[::-1].reset_index(drop=True)
    # print(ind, ind[1] - ind[0])
    
    demands.append(data['Grid [kW]'].values)
    # print(data)
demands = np.sum(demands, axis = 0)
# print(min(demands), max(demands))
##-73.26824166700004 616.9410838889999

datetimes = data['Date & Time'].values
temp = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in datetimes]
Month, Day, Hour, Weekday = [], [], [], []
for i in range(len(temp)):
    Month.append(temp[i].month)
    Weekday.append(int(temp[i].isoweekday() in range(1, 6)))
    Day.append(temp[i].day)
    Hour.append(temp[i].hour)
Date_dic = {'Month':Month, 'Day':Day, 'Weekday':Weekday, 'Hour':Hour}
Date = pd.DataFrame(Date_dic)

Dataframe = pd.read_csv(mypath1)
ind1 = Dataframe.index[Dataframe['REPORT_TYPE'] == 'FM-15'].tolist()
Dataframe = Dataframe.iloc[ind1[4:-3]].reset_index(drop=True)
# Dataframe['HourlyDewPointTemperature'] = Dataframe['HourlyDewPointTemperature'].apply(lambda x: (x-32)*0.5556 )

Date['Temperature'] = pd.to_numeric(Dataframe['HourlyDewPointTemperature']).values
Date['Demand'] = demands
Date.to_csv('data.csv',index=False)