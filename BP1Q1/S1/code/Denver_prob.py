# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 20:41:57 2021

@author: Junkai
"""

import pandas as pd
import numpy as np
from util import data_loader
from pf import QR_pf, GB_pf, RF_pf, LSTM_pf, GRU_pf, DNN_pf

name = '../data_cleaned/data.csv'
data, minimum, maximum = data_loader(name)
Results, Scores = pd.DataFrame(), pd.DataFrame()
Methods = ['QR', 'GB', 'RF', 'LSTM', 'DNN', 'GRU']
quantiles = np.arange(1,10)*0.1

Train, Test = data[:-(31+30)*24+1].reset_index(drop=True), data[-(31+30)*24+1:].reset_index(drop=True)
Y_train, Y_test = Train['Demand'].values, Test['Demand'].values
X_train, X_test = Train.drop(columns=['Demand']).values, Test.drop(columns=['Demand']).values

Y_test = Y_test * (maximum - minimum) + minimum

QR_pred, QR_Pinball, QR_mean = QR_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
GB_pred, GB_Pinball, GB_mean = GB_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
RF_pred, RF_Pinball, RF_mean = RF_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
LSTM_pred, LSTM_Pinball, LSTM_mean = LSTM_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
GRU_pred, GRU_Pinball, GRU_mean = GRU_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
DNN_pred, DNN_Pinball, DNN_mean = DNN_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)

print( "QR: {:.4f}, GB: {:.4f}, RF: {:.4f}, LSTM: {:.4f}, DNN: {:.4f}, GRU: {:.4f}".format(QR_mean,\
                                                                                           GB_mean, RF_mean, LSTM_mean, GRU_mean, DNN_mean))
    
Results['True'] = Y_test

for j, tau in enumerate(quantiles):
    Results[Methods[0]+'-'+str(tau)] = QR_pred[j, :]
    Results[Methods[1]+'-'+str(tau)] = GB_pred[j, :]
    Results[Methods[2]+'-'+str(tau)] = RF_pred[j, :]
    Results[Methods[3]+'-'+str(tau)] = LSTM_pred[j, :]
    Results[Methods[4]+'-'+str(tau)] = DNN_pred[j, :]
    Results[Methods[5]+'-'+str(tau)] = GRU_pred[j, :]

Scores[Methods[0]], Scores[Methods[1]], Scores[Methods[2]], Scores[Methods[3]], Scores[Methods[4]], Scores[Methods[5]], = \
    [QR_mean], [GB_mean], [RF_mean], [LSTM_mean], [DNN_mean], [GRU_mean]
    
Results.to_csv('Results_prob.csv',index=False)
Scores.to_csv('Scores_prob.csv',index=False)
