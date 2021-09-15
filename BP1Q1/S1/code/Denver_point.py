# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 17:27:57 2021

@author: Junkai
"""

import pandas as pd
import numpy as np
from util import data_loader, LR_base, DT_base, SVM_base, LSTM_base, GRU_base, DNN_base

name = '../data_cleaned/data.csv'
data, minimum, maximum = data_loader(name)
Results, Scores = pd.DataFrame(), pd.DataFrame()

#Test the last two months
Train, Test = data[:-(31+30)*24+1].reset_index(drop=True), data[-(31+30)*24+1:].reset_index(drop=True)
Y_train, Y_test = Train['Demand'].values, Test['Demand'].values
X_train, X_test = Train.drop(columns=['Demand']).values, Test.drop(columns=['Demand']).values


LR_pred, LR_mape, LR_mae = LR_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
DT_pred, DT_mape, DT_mae = DT_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
SVR_pred, SVR_mape, SVR_mae = DT_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
LSTM_pred, LSTM_mape, LSTM_mae = LSTM_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
DNN_pred, DNN_mape, DNN_mae = DNN_base(Y_train, Y_test, X_train, X_test, minimum, maximum)
GRU_pred, GRU_mape, GRU_mae = GRU_base(Y_train, Y_test, X_train, X_test, minimum, maximum)


print(LR_mape, DT_mape, SVR_mape, LSTM_mape, DNN_mape, GRU_mape)
print(LR_mae, DT_mae, SVR_mae, LSTM_mae, DNN_mae, GRU_mape)

Y_test = Y_test * (maximum - minimum) + minimum

Results['True'] = Y_test
Results['LR'], Results['DT'], Results['SVR'], Results['LSTM'], Results['DNN'], Results['GRU'] =\
    LR_pred, DT_pred, SVR_pred, LSTM_pred, DNN_pred, GRU_pred
    
Scores['LR_mape'], Scores['DT_mape'], Scores['SVR_mape'], Scores['LSTM_mape'], Scores['DNN_mape'], Scores['GRU_mape'] = \
    [LR_mape], [DT_mape], [SVR_mape], [LSTM_mape], [DNN_mape], [GRU_mape]
    
Scores['LR_mae'], Scores['DT_mae'], Scores['SVR_mae'], Scores['LSTM_mae'], Scores['DNN_mae'], Scores['GRU_mae'] = \
    [LR_mae], [DT_mae], [SVR_mae], [LSTM_mae], [DNN_mae], [GRU_mape]
    
Results.to_csv('../result/Results.csv',index=False)
Scores.to_csv('../result/Scores.csv',index=False)
