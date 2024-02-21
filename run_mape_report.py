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
    full_scale = abs(target).max(axis=0)

    error = target - predict
    error_percent = error / full_scale * 100

    return abs(error_percent).mean(axis=0)


def read_csv(filename):
    return pd.read_csv(filename, index_col='date', parse_dates=['date'])


all_mape = list()

for predict_filename, target_filename in zip(args.predict, args.target):
    print(predict_filename, target_filename)

    predict = read_csv(predict_filename)
    target = read_csv(target_filename)

    mape = calculate_mape(predict=predict, target=target).to_frame()
    all_mape.append(mape)

report = pd.concat(all_mape, axis=1).T
report.insert(0, 'target', [str(filename) for filename in args.target])
report.insert(0, 'predict', [str(filename) for filename in args.predict])

report.set_index('predict', inplace=True)
report.sort_index(inplace=True)
report.to_csv(args.save)
