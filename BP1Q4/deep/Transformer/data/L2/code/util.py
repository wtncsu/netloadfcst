# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:11:10 2021

@author: Junkai
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
seed_value= 0
# set_random_seed(seed_value)#fix later
np.random.seed(seed_value)

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / max(y_true))) * 100

def data_loader(name):
    cd = '../data_cleaned/'
    Dates = pd.read_csv(cd+'Date.csv')
    demand_data = pd.read_csv(cd+name+'.csv')
    maximum, minimum = max(demand_data['Net']), min(demand_data['Net'])
    demand_data['Demand'] = demand_data['Net'].apply(lambda x: (x - minimum) / (maximum - minimum) )
    Data = Dates.copy()
    Data[['Demand', 'Temperature']] = demand_data[['Demand', 'Temperature']]
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

def LSTM_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    _, num_features = X_train.shape
    X_train, X_test = X_train.reshape(-1, 1, num_features), X_test.reshape(-1, 1, num_features)
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(32, activation='relu'))
    model.add((Dense(1, activation='linear')))
    model.compile(loss="mean_absolute_error", optimizer="rmsprop")
    model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=0)
    LSTM_pred = model.predict(X_test)
    Y_test, LSTM_pred = Y_test * (maximum - minimum) + minimum, LSTM_pred * (maximum - minimum) + minimum
    return LSTM_pred, mape(Y_test.reshape(-1), LSTM_pred.reshape(-1)), mae(Y_test.reshape(-1), LSTM_pred.reshape(-1))
# Y_train, Y_test = Y_train.reshape(-1, 24, 1), Y_test.reshape(-1, 24, 1)
# model = Sequential()
# model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
# # model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(TimeDistributed(Dense(1, activation='linear')))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")
# model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.2)
# predictions = model.predict(X_test)
# print(mape(Y_test.reshape(-1),predictions.reshape(-1)))

def DNN_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="rmsprop")
    model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_split=0.1, verbose=0)
    DNN_pred = model.predict(X_test)
    Y_test, DNN_pred = Y_test * (maximum - minimum) + minimum, DNN_pred * (maximum - minimum) + minimum
    return DNN_pred, mape(Y_test, DNN_pred), mae(Y_test, DNN_pred)

def GRU_base(Y_train, Y_test, X_train, X_test, minimum, maximum):
    _, num_features = X_train.shape
    X_train, X_test = X_train.reshape(-1, 1, num_features), X_test.reshape(-1, 1, num_features)
    model = Sequential()
    model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(32, activation='relu'))
    model.add((Dense(1, activation='linear')))
    model.compile(loss="mean_absolute_error", optimizer="rmsprop")
    model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=0)
    GRU_pred = model.predict(X_test)
    Y_test, GRU_pred = Y_test * (maximum - minimum) + minimum, GRU_pred * (maximum - minimum) + minimum
    return GRU_pred, mape(Y_test.reshape(-1), GRU_pred.reshape(-1)), mae(Y_test.reshape(-1), GRU_pred.reshape(-1))

if __name__ == "__main__":
    names = ['COAST', 'EAST', 'FWEST', 'NCENT', 'NORTH', 'SCENT', 'SOUTH', 'WEST']
    name = names[0]
    Data, minimum, maximum = data_loader(name)
    
