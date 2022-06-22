# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:33:08 2021

@author: jliang9
"""
import numpy as np
from util import mape, mae
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from tensorflow import set_random_seed

seed_value= 0
set_random_seed(seed_value)#fix later
np.random.seed(seed_value)

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