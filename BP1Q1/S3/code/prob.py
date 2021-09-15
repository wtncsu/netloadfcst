# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 00:04:58 2021

@author: jliang9

Baselines for probabilistic forecasting
"""

import pandas as pd
import numpy as np
from util import data_loader
from pf import QR_pf, GB_pf, RF_pf, LSTM_pf, GRU_pf, DNN_pf


names = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DK', 'ES', 'FR', 'GR', 'IT', 'NL', 'PT', 'SI', 'SK']
Methods = ['QR', 'GB', 'RF', 'LSTM', 'DNN', 'GRU']
Results, Scores = pd.DataFrame(), pd.DataFrame()
quantiles = np.arange(1,10)*0.1

for i in range(len(names)):
    data, minimum, maximum = data_loader(names[i])
    #use the recent 3 years 365, 366, 365, 365, 365
    data = data.iloc[(365+366)*24:].reset_index(drop=True)
    #First two years are used for training, and the last year is for testing
    Train, Test = data[:(365+365)*24].reset_index(drop=True), data[(365+365)*24:].reset_index(drop=True)
    
    Y_train, Y_test = Train['Demand'].values, Test['Demand'].values
    X_train, X_test = Train.drop(columns=['Demand']).values, Test.drop(columns=['Demand']).values
    Y_test = Y_test * (maximum - minimum) + minimum
    
    QR_pred, QR_Pinball, QR_mean = QR_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    GB_pred, GB_Pinball, GB_mean = GB_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    RF_pred, RF_Pinball, RF_mean = RF_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    LSTM_pred, LSTM_Pinball, LSTM_mean = LSTM_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    GRU_pred, GRU_Pinball, GRU_mean = GRU_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    DNN_pred, DNN_Pinball, DNN_mean = DNN_pf(Y_train, Y_test, X_train, X_test, minimum, maximum, quantiles)
    
    print("Zone " + names[i])
    print( "QR: {:.4f}, GB: {:.4f}, RF: {:.4f}, LSTM: {:.4f}, DNN: {:.4f}, GRU: {:.4f}".format(QR_mean, GB_mean, RF_mean, LSTM_mean, GRU_mean, DNN_mean))
    
    Results[names[i]+' True'] = Y_test
    for j, tau in enumerate(quantiles):
        Results[names[i]+': '+Methods[0]+'-'+str(tau)] = QR_pred[j, :]
        Results[names[i]+': '+Methods[1]+'-'+str(tau)] = GB_pred[j, :]
        Results[names[i]+': '+Methods[2]+'-'+str(tau)] = RF_pred[j, :]
        Results[names[i]+': '+Methods[3]+'-'+str(tau)] = LSTM_pred[j, :]
        Results[names[i]+': '+Methods[4]+'-'+str(tau)] = DNN_pred[j, :]
        Results[names[i]+': '+Methods[5]+'-'+str(tau)] = GRU_pred[j, :]
    
    Scores[names[i]+': '+Methods[0]], Scores[names[i]+': '+Methods[1]], Scores[names[i]+': '+Methods[2]],\
        Scores[names[i]+': '+Methods[3]], Scores[names[i]+': '+Methods[4]],Scores[names[i]+': '+Methods[5]] = \
            [QR_mean], [GB_mean], [RF_mean], [LSTM_mean], [DNN_mean], [GRU_mean]

Results.to_csv('../result/Results_prob.csv',index=False)
Scores.to_csv('../result/Scores_prob.csv',index=False)

