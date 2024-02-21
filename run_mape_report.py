#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()

parser.add_argument('--predict', type=Path, required=True, action='append')
parser.add_argument('--target', type=Path, required=True, action='append')
parser.add_argument('--save', type=Path, required=True)
args = parser.parse_args()

if len(args.predict) != len(args.target):
    raise ValueError(f'Mismatched number of predict and target datasets.')


def calculate_mape(predict, target):
    predict = predict.ravel()
    target = target.ravel()
    full_scale = abs(target).max()

    error = target - predict
    error_percent = error / full_scale * 100

    return abs(error_percent).mean()


def read_csv(filename):
    return pd.read_csv(filename, index_col='date', parse_dates=['date'])


all_mape = list()

for predict_filename, target_filename in zip(args.predict, args.target):
    predict = read_csv(predict_filename).to_numpy()
    target = read_csv(target_filename).to_numpy()

    mape = calculate_mape(predict=predict, target=target)
    all_mape.append(mape)

report = pd.DataFrame({
    'target':  [str(filename) for filename in args.target],
    'predict': [str(filename) for filename in args.predict],
    'mape':    all_mape
})

report.set_index('predict', inplace=True)
report.sort_index(inplace=True)
report.to_csv(args.save)
