# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:24:45 2020

@author: Junkai
"""
from datetime import datetime
import numpy as np
import pandas as pd

demand_data = pd.read_csv('demand_data.csv').fillna(0)
weather_data = pd.read_csv('weather_data.csv').fillna(0)
names = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DK', 'ES', 'FR', 'GR', 'IT', 'NL', 'PT', 'SI', 'SK']
names2 = ['_load_actual_entsoe_transparency', '_solar_generation_actual', '_temperature', '_radiation_direct_horizontal', '_radiation_diffuse_horizontal']
#k = [2, 3, 1.3, 2, 2.2, 1, 8, 16, 1.8, 10, 4.4, 1.8, 1, 2]
# This can be used to adjust the power of each country, if needed. The magnitude is not in the same level. 
k = 14 *[1.]
for i in range(len(k)):
    demand_data[names[i]+names2[0]] = demand_data[names[i]+names2[0]].apply(lambda x: x / k[i])
    demand_data[names[i]+names2[1]] = demand_data[names[i]+names2[1]].apply(lambda x: x / k[i])
    
feature_dates = demand_data['utc_timestamp'].values
temp = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in feature_dates]
Month, Day, Hour, Weekday = [], [], [], []
for i in range(len(temp)):
    Month.append(temp[i].month)
    Weekday.append(int(temp[i].isoweekday() in range(1, 6)))
    Day.append(temp[i].day)
    Hour.append(temp[i].hour)
Date_dic = {'Month':Month, 'Day':Day, 'Weekday':Weekday, 'Hour':Hour}
Date = pd.DataFrame(Date_dic)
Date.to_csv('Date.csv',index=False)  


maximum_demand, mini_demand = [], []  
for s in names:
    demand = demand_data[s+names2[0]].values - demand_data[s+names2[1]].values
    a, b, c = max(demand), min(demand), max(demand_data[s+names2[1]].values)
    maximum_demand.append(a)
    mini_demand.append(b)
    
    features = pd.DataFrame()
    features['Demand'], features['Normalized_net'] = demand, demand
    features['Normalized_net'] = features['Normalized_net'].apply(lambda x: x - b )
    features['Normalized_net'] = features['Normalized_net'].apply(lambda x: x / (a - b))
    features['Temperature'], features['DNI'], features['DHI'] = weather_data[s+names2[2]],\
        weather_data[s+names2[3]], weather_data[s+names2[4]]
    features.to_csv(s+'.csv',index=False)
    #print(max(features['Normalized_net'].values), min(features['Normalized_net'].values))
    print(a, b, c/a)

    
'''
10799.0 664.0 0.09037873877210853
13670.0 5065.82 0.19809948792977322
7690.0 -588.0 0.11131339401820546
18529.620000000003 0.0 0.056453936993850914
10893.41 -1057.97 0.15819472506772442
8604.029999999999 0.0 0.09321329655986789
40912.0 -3836.0 0.16493938208838482
94306.0 -5342.0 0.07111954700655314
9438.38 -1397.0 0.21846969501122016
55157.0 0.0 0.19257755135340937
19272.0 6483.0 0.11737235367372353
8732.2 0.0 0.059183252788529796
2402.19 -43.44000000000001 0.2988189943343366
17492.6 -235.2 0.02213507425997279
'''


