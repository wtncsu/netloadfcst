# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:42:56 2021

@author: Junkai
"""
from datetime import datetime
import numpy as np
import pandas as pd

Names = ['COAST', 'EAST', 'FWEST', 'NCENT', 'NORTH', 'SCENT', 'SOUTH', 'WEST']
Years = ['_2015.csv', '_2016.csv', '_2017.csv', '_2018.csv', '_2019.csv']


# datetimes = pd.date_range('2015-01-01', periods=(365+366+365+365+365)*24, freq='60T')
# datetimes = np.datetime_as_string(datetimes, unit='h')
# temp = [datetime.strptime(t, "%Y-%m-%dT%H") for t in datetimes]
# Month, Day, Hour, Weekday = [], [], [], []
# for i in range(len(temp)):
#     Month.append(temp[i].month)
#     Weekday.append(int(temp[i].isoweekday() in range(1, 6)))
#     Day.append(temp[i].day)
#     Hour.append(temp[i].hour)
# Date_dic = {'Month':Month, 'Day':Day, 'Weekday':Weekday, 'Hour':Hour}
# Date = pd.DataFrame(Date_dic)
# Date.to_csv('Date.csv',index=False)  

for i in range(len(Names)):
    for j in range(len(Years)):
        if j == 0:
            data = pd.read_csv(Names[i]+Years[j]).fillna(0)
        else:
            data = pd.concat([data, pd.read_csv(Names[i]+Years[j]).fillna(0)], axis = 0)
    data = data.reset_index(drop=True)
    
    data['Net'] = data['Load(MW)'] - data['PV(MW)']
    data['Temperature'] = data['Ambient Temperature(C)']
    a, b, c = max(data['Load(MW)']), min((data['Load(MW)'])), max(data['PV(MW)'])
    print(a, b, c)
    data = data[['Net', 'Temperature']]
    data.to_csv(Names[i]+'.csv',index=False)
    
'''
21256.11519 0.0 4676.345341800001
2620.839043 0.0 561.8833833799998
4307.552925 0.0 947.6616435000001
26499.240955 0.0 5608.63410008
1521.845309 0.0 324.79258218
14166.701622999999 0.0 2833.3403246000007
6176.106417 0.0 1328.7369609799996
2117.399078 0.0 465.8277971600001
'''