# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:16:21 2021

@author: Junkai
"""

import pandas as pd
import numpy as np
from util import data_loader, mape, mae, LR_base, DT_base, SVM_base, LSTM_base, DNN_base, GRU_base


names = ['MIDATL', 'SOUTH', 'WEST']
Methods = ['LR', 'DT', 'SVR', 'LSTM', 'DNN', 'GRU']
Results, Scores = pd.DataFrame(), pd.DataFrame()

for i in range(len(names)):
    data, minimum, maximum = data_loader(names[i])

    #First 1.5 years are used for training, and the last 0.5 year is for testing
    Train, Test = data[:(365+366-183)*24].reset_index(drop=True), data[(365+366-183)*24:].reset_index(drop=True)
    
    Y_train, Y_test = Train['Demand'].values, Test['Demand'].values
    X_train, X_test = Train.drop(columns=['Demand']).values, Test.drop(columns=['Demand']).values
    #print(len(Y_train), len(Y_test), X_train.shape, X_test.shape)
    
    LR_pred, LR_mape, LR_mae = LR_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    DT_pred, DT_mape, DT_mae = DT_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    SVR_pred, SVR_mape, SVR_mae = DT_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    LSTM_pred, LSTM_mape, LSTM_mae = LSTM_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    DNN_pred, DNN_mape, DNN_mae = DNN_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    GRU_pred, GRU_mape, GRU_mae = GRU_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
    
    Y_test = Y_test * (maximum - minimum) + minimum
    Results[names[i]+' True'] = Y_test
    Results[names[i]+' '+Methods[0]], Results[names[i]+' '+Methods[1]], Results[names[i]+' '+Methods[2]], Results[names[i]+' '+Methods[3]],\
        Results[names[i]+' '+Methods[4]], Results[names[i]+' '+Methods[5]] = LR_pred, DT_pred, SVR_pred, LSTM_pred, DNN_pred, GRU_pred
    Scores[names[i]+' '+'LR_mape'], Scores[names[i]+' '+'DT_mape'], Scores[names[i]+' '+'SVR_mape'], Scores[names[i]+' '+'LSTM_mape'], Scores[names[i]+' '+'DNN_mape'],\
        Scores[names[i]+' '+'GRU_mape'] = [LR_mape], [DT_mape], [SVR_mape], [LSTM_mape], [DNN_mape], [GRU_mape]
    Scores[names[i]+' '+'LR_mae'], Scores[names[i]+' '+'DT_mae'], Scores[names[i]+' '+'SVR_mae'], Scores[names[i]+' '+'LSTM_mae'], Scores[names[i]+' '+'DNN_mae'], \
        Scores[names[i]+' '+'GRU_mae'] = [LR_mae], [DT_mae], [SVR_mae], [LSTM_mae], [DNN_mae], [GRU_mae]
    
    print("Zone " + names[i])
    print( "LR_mape: {:.4f}, DT_mape: {:.4f}, SVR_mape: {:.4f}, LSTM_mape: {:.4f}, DNN_mape: {:.4f}, GRU_mape: {:.4f}"\
          .format(LR_mape, DT_mape, SVR_mape, LSTM_mape, DNN_mape, GRU_mape))
    print( "LR_mae: {:.4f}, DT_mae: {:.4f}, SVR_mae: {:.4f}, LSTM_mae: {:.4f}, DNN_mae: {:.4f}, GRU_mae: {:.4f}"\
          .format(LR_mae, DT_mae, SVR_mae, LSTM_mae, DNN_mae, GRU_mae))
      
      
Results.to_csv('../result/Results.csv',index=False)
Scores.to_csv('../result/Scores.csv',index=False)
