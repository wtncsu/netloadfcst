# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 00:47:47 2021

@author: jliang9
Methods for probabilistic forecasting
"""
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import keras.backend as K
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
# from tensorflow import set_random_seed

seed_value= 0
# set_random_seed(seed_value)#fix later
np.random.seed(seed_value)

def Pinball(yhat, tau, y):
    if yhat >= y:
        score = (1-tau) * (yhat - y)
    else:
        score = tau * (y - yhat)
    return score

def QR_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
     nstep = len(Y_test)
     QR_pred = np.array([QuantReg(Y_train, X_train).fit(q=i).predict(X_test) for i in quantiles])
     QR_pred = QR_pred * (maximum - minimum) + minimum
     QR_Pinball = np.array([np.mean([Pinball(QR_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
     return QR_pred, QR_Pinball, np.mean(QR_Pinball)
 
def GB_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
     nstep = len(Y_test)
     GB_pred = np.array([GradientBoostingRegressor(loss='quantile', alpha=i).fit(X_train, Y_train).predict(X_test) for i in quantiles])
     GB_pred = GB_pred * (maximum - minimum) + minimum
     GB_Pinball = np.array([np.mean([Pinball(GB_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
     return GB_pred, GB_Pinball, np.mean(GB_Pinball)
 
def RF_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
     nstep = len(Y_test)
     RF = RandomForestRegressor(random_state=0, n_estimators = 100)
     RF.fit(X_train, Y_train)
     def rf_quantile(m, X, q):
         # https://towardsdatascience.com/quantile-regression-from-linear-models-to-trees-to-deep-learning-af3738b527c3
         # m: sklearn random forests model.
         # X: X matrix.
         # q: Quantile.
         rf_preds = []
         for estimator in m.estimators_:
             rf_preds.append(estimator.predict(X))
         rf_preds = np.array(rf_preds).transpose()
         return np.percentile(rf_preds, q * 100, axis=1)
     RF_pred = rf_quantile(RF, X_test, quantiles)
     RF_pred = RF_pred * (maximum - minimum) + minimum
     RF_Pinball = np.array([np.mean([Pinball(RF_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
     return RF_pred, RF_Pinball, np.mean(RF_Pinball)
 
def LSTM_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
    LSTM_pred, nstep = [], len(Y_test)
    _, num_features = X_train.shape
    X_train, X_test = X_train.reshape(-1, 1, num_features), X_test.reshape(-1, 1, num_features)
    for tau in quantiles:
        def pinball_loss(y_true, y_pred):
            err = y_true - y_pred
            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)
        model = Sequential()
        model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(32, activation='relu'))
        model.add((Dense(1, activation='linear')))
        model.compile(loss=pinball_loss, optimizer="rmsprop")
        model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=0)
        LSTM_pred.append(model.predict(X_test).reshape(-1))
    LSTM_pred = np.array(LSTM_pred) * (maximum - minimum) + minimum   
    LSTM_Pinball = np.array([np.mean([Pinball(LSTM_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
    return LSTM_pred, LSTM_Pinball, np.mean(LSTM_Pinball)
 
def GRU_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
    GRU_pred, nstep = [], len(Y_test)
    _, num_features = X_train.shape
    X_train, X_test = X_train.reshape(-1, 1, num_features), X_test.reshape(-1, 1, num_features)
    for tau in quantiles:
        def pinball_loss(y_true, y_pred):
            err = y_true - y_pred
            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)
        model = Sequential()
        model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(32, activation='relu'))
        model.add((Dense(1, activation='linear')))
        model.compile(loss=pinball_loss, optimizer="rmsprop")
        model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=0)
        GRU_pred.append(model.predict(X_test).reshape(-1))
    GRU_pred = np.array(GRU_pred) * (maximum - minimum) + minimum   
    GRU_Pinball = np.array([np.mean([Pinball(GRU_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
    return GRU_pred, GRU_Pinball, np.mean(GRU_Pinball)
 
def DNN_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles):
    DNN_pred, nstep = [], len(Y_test)
    for tau in quantiles:
        def pinball_loss(y_true, y_pred):
            err = y_true - y_pred
            return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=pinball_loss, optimizer="rmsprop")
        model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.1, verbose=0)
        DNN_pred.append(model.predict(X_test).reshape(-1))
    DNN_pred = np.array(DNN_pred) * (maximum - minimum) + minimum   
    DNN_Pinball = np.array([np.mean([Pinball(DNN_pred[i, t], tau, Y_test[t]) for t in range(nstep)]) for i, tau in enumerate(quantiles)])
    return DNN_pred, DNN_Pinball, np.mean(DNN_Pinball)
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    