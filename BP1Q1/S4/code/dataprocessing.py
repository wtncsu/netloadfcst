# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:42:56 2021

@author: Junkai
"""
from datetime import datetime
import numpy as np
import pandas as pd

Names = ['MIDATL', 'SOUTH', 'WEST']
Years = ['_2019.csv', '_2020.csv']


datetimes = pd.date_range('2019-01-01', periods=(365+366)*24, freq='60T')
datetimes = np.datetime_as_string(datetimes, unit='h')
temp = [datetime.strptime(t, "%Y-%m-%dT%H") for t in datetimes]
Month, Day, Hour, Weekday = [], [], [], []
for i in range(len(temp)):
    Month.append(temp[i].month)
    Weekday.append(int(temp[i].isoweekday() in range(1, 6)))
    Day.append(temp[i].day)
    Hour.append(temp[i].hour)
Date_dic = {'Month':Month, 'Day':Day, 'Weekday':Weekday, 'Hour':Hour}
Date = pd.DataFrame(Date_dic)
Date.to_csv('Date.csv',index=False)  

k = 5
for i in range(len(Names)):
    for j in range(len(Years)):
        if j == 0:
            data = pd.read_csv(Names[i]+Years[j]).fillna(0)
        else:
            data = pd.concat([data, pd.read_csv(Names[i]+Years[j]).fillna(0)], axis = 0)
    data = data.reset_index(drop=True)
    data['PV(MW)'] = data['PV(MW)'].apply(lambda x: x * k )
    data['Net'] = data['Load(MW)'] - data['PV(MW)']
    data['Temperature'] = data['Ambient Temperature (C)']
    a, b, c = max(data['Load(MW)']), min((data['Load(MW)'])), max(data['PV(MW)'])
    print(a, b, c)
    data = data[['Net', 'Temperature']]
    data.to_csv(Names[i]+'.csv',index=False)
    
'''
56585.042 18518.006999999998 787.953
20086.521 7245.522 1254.264
76613.86200000001 28917.465 212.709
'''