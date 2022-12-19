#!/usr/bin/env python3
from pandas import read_csv, DataFrame
from configparser import ConfigParser
from argparse import ArgumentParser
from pathlib import Path
from time import time
from ngboost import NGBRegressor

aparser = ArgumentParser(
    description='Fit and predict dataset using NGboost decision tree'
)
aparser.add_argument('config', type=Path)
aparser.add_argument('--datadir', type=Path, default='output')
aparser.add_argument('--outdir', type=Path, default='output')
args = aparser.parse_args()

args.outdir.mkdir(parents=True, exist_ok=True)

configs = ConfigParser()
configs.read(args.config)

output_name = args.config.stem

train_features_path = args.datadir / f'train-features-{output_name}.csv'
train_target_path = args.datadir / f'train-target-{output_name}.csv'
test_features_path = args.datadir / f'test-features-{output_name}.csv'
train_pred_mean_path = args.outdir / f'obdt-train-pred-mean-{output_name}.csv'
train_pred_std_path = args.outdir/ f'obdt-train-pred-std-{output_name}.csv'
test_pred_mean_path = args.outdir / f'obdt-test-pred-mean-{output_name}.csv'
test_pred_std_path = args.outdir / f'obdt-test-pred-std-{output_name}.csv'
timing_path = args.outdir / f'obdt-timing-{output_name}.csv'

df_train_features = read_csv(train_features_path, index_col=0)
df_train_target = read_csv(train_target_path, index_col=0)
df_test_features = read_csv(test_features_path, index_col=0)

train_features = df_train_features.to_numpy()
train_target   = df_train_target.to_numpy()
test_features  = df_test_features.to_numpy()

train_index = df_train_features.index
test_index = df_test_features.index

model = NGBRegressor()

fit_start = time()
model.fit(train_features, train_target)
fit_end = time()

train_pred_start = time()
train_pred = model.pred_dist(train_features)
train_pred_end = time()

test_pred_start = time()
test_pred = model.pred_dist(test_features)
test_pred_end = time()

df_train_pred_mean = DataFrame(
    train_pred.params['loc'], columns=['mean'], index=train_index)

df_train_pred_std = DataFrame(
    train_pred.params['scale'], columns=['std'], index=train_index)

df_test_pred_mean = DataFrame(
    test_pred.params['loc'], columns=['mean'], index=test_index)

df_test_pred_std = DataFrame(
    test_pred.params['scale'], columns=['std'], index=test_index)

df_train_pred_mean.to_csv(train_pred_mean_path)
df_train_pred_std.to_csv(train_pred_std_path)
df_test_pred_mean.to_csv(test_pred_mean_path)
df_test_pred_std.to_csv(test_pred_std_path)

fit_time = fit_end - fit_start
train_pred_time = train_pred_end - train_pred_start
test_pred_time = test_pred_end - test_pred_start

timing = DataFrame(
    [[fit_time, train_pred_time, test_pred_time]],
    columns=['fit', 'train_pred', 'test_pred'],
    index=['obdt']
)

timing.to_csv(timing_path)
