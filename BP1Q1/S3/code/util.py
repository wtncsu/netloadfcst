# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:54:21 2021

@author: jliang9

util
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / max(y_true))) * 100

def data_loader(name):
    cd = '../data_cleaned/'
    Dates = pd.read_csv(cd+'Date.csv')
    demand_data = pd.read_csv(cd+name+'.csv')[['Demand', 'Normalized_net', 'Temperature']]
    maximum, minimum = max(demand_data['Demand']), min(demand_data['Demand'])
    Data = Dates.copy()
    Data[['Demand', 'Temperature']] = demand_data[['Normalized_net', 'Temperature']]
    Data['Past1day'] = Data['Demand'].shift(periods=24).fillna(0)
    Data['Past1week'] = Data['Demand'].shift(periods=24*7).fillna(0)
    #Create temperature squred, cubed 
    # Data['Temperature2'], Data['Temperature3'] = Data['Temperature'].apply(lambda x: x ** 2 ), Data['Temperature'].apply(lambda x: x ** 3 )
    # #Create some cross terms. The selection is based on Tao's Vanilla model. 
    # Data['WH'], Data['TH'], Data['TH2'], Data['TH3'] = Data['Weekday'] * Data['Hour'], Data['Temperature'] * Data['Hour'], \
    #     Data['Temperature2'] * Data['Hour'], Data['Temperature3'] * Data['Hour']
    # Data['TM'], Data['TM2'], Data['TM3'] =  Data['Temperature'] * Data['Month'], Data['Temperature2'] * Data['Month'], Data['Temperature3'] * Data['Month']
    #Those terms are commented out, it provides very limited improvement except for LR.
    return Data, minimum, maximum

def LR_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    LR = LinearRegression().fit(X_train, Y_train)
    # LR = Ridge().fit(X_train, Y_train)
    # LR = Lasso().fit(X_train, Y_train)
    LR_pred = LR.predict(X_test)
    Y_test, LR_pred = Y_test * (maximum - minimum) + minimum, LR_pred * (maximum - minimum) + minimum
    return LR_pred, mape(Y_test, LR_pred), mae(Y_test, LR_pred)

def DT_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    #default=”mse”
    regr = DecisionTreeRegressor().fit(X_train, Y_train)
    # regr = RandomForestRegressor(random_state=0).fit(X_train, Y_train)
    DT_pred = regr.predict(X_test)
    Y_test, DT_pred = Y_test * (maximum - minimum) + minimum, DT_pred * (maximum - minimum) + minimum
    return DT_pred, mape(Y_test, DT_pred), mae(Y_test, DT_pred)

def SVM_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    #default=’rbf’
    regr = svm.SVR(kernel='rbf', gamma='scale').fit(X_train, Y_train)
    SVR_pred = regr.predict(X_test)
    Y_test, SVR_pred = Y_test * (maximum - minimum) + minimum, SVR_pred * (maximum - minimum) + minimum
    return SVR_pred, mape(Y_test, SVR_pred), mae(Y_test, SVR_pred)
   
if __name__ == "__main__":
    names = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DK', 'ES', 'FR', 'GR', 'IT', 'NL', 'PT', 'SI', 'SK']
    name = names[0]
    Data, minimum, maximum = data_loader(name)
    
    # print(minimum, maximum)
    # print(Data)
    
    
    
    
    
    
    
    
    