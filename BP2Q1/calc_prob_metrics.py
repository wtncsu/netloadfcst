#!/usr/bin/env python3
import numpy
from pandas import DataFrame, read_csv
from argparse import ArgumentParser
from pathlib import Path
from scipy.stats import norm
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
from pydatools.ngboost.distributions import Normal_Parameters
from pydatools.ngboost.scoring_rules import CRPS

aparser = ArgumentParser(
    description='Calculate probabilistic forecast accuracy'
)

aparser.add_argument('--mean', type=Path, required=True)
aparser.add_argument('--std', type=Path, required=True)
aparser.add_argument('--target', type=Path, required=True)
aparser.add_argument('--out', type=Path, default=None)
args = aparser.parse_args()

mean = read_csv(args.mean, index_col='time').to_numpy().ravel()
std = read_csv(args.std, index_col='time').to_numpy().ravel()
target = read_csv(args.target, index_col='time').to_numpy().ravel()

point_metrics = {
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

percentiles = numpy.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def mean_pinball_loss(pred, target):
    tau = percentiles
    max1 = (target - pred).clip(min=0)
    max2 = (pred - target).clip(min=0)
    loss = (tau * max1 + (1 - tau) * max2).mean()
    return loss


df = DataFrame({
    name: func(mean, target)
    for name, func in point_metrics.items()
}, index=[0])

logstd = numpy.log(std)
dist = Normal_Parameters(mean, logstd)
df['mean_crps'] = CRPS.primitive(dist, target).mean()

normal_dist = dist.to_distribution()
pinball_pred = norm.ppf(percentiles,
                        loc=mean.reshape(-1, 1), scale=std.reshape(-1, 1))


df[f'mean_pinball'] = mean_pinball_loss(pinball_pred, target.reshape(-1, 1))

out_file_path = args.out if args.out is not None else (
    args.target.parent / f'{args.target.stem}-metrics.csv')

df.to_csv(out_file_path, index=False)

