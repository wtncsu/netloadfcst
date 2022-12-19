#!/usr/bin/env python3
from pandas import DataFrame, read_csv
from argparse import ArgumentParser
from pathlib import Path
from pydatools.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    sum_of_squared_error,
    mean_absolute_percent_error,
    mean_absolute_percent_full_scale_error,
    weighted_mean_absolute_percent_error,
    mean_bias_error,
    coefficient_of_determination,
)

aparser = ArgumentParser(
    description='Calculate point forecast accuracy'
)

aparser.add_argument('--pred', type=Path, required=True)
aparser.add_argument('--target', type=Path, required=True)
aparser.add_argument('--out', type=Path, default=None)
args = aparser.parse_args()

metrics = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'err_sq': sum_of_squared_error,
    'mape': mean_absolute_percent_error,
    'mape_fs': mean_absolute_percent_full_scale_error,
    'wmape': weighted_mean_absolute_percent_error,
    'bias_err': mean_bias_error,
    'r_sq': coefficient_of_determination,
}

pred = read_csv(args.pred, index_col=0).to_numpy()
target = read_csv(args.target, index_col=0).to_numpy()

df = DataFrame({
    name: func(pred, target)
    for name, func in metrics.items()
}, index=[0])

out_file_path = args.out if args.out is not None else (
    args.target.parent / f'{args.target.stem}-metrics.csv')

df.to_csv(out_file_path, index=False)
