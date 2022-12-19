#!/usr/bin/env python3
from pandas import read_csv, DataFrame
from configparser import ConfigParser
from argparse import ArgumentParser
from pathlib import Path
from time import time
from pydatools.regression import (
    Fuzzy_Decision_Tree_Regressor,
)

aparser = ArgumentParser(
    description='Fit and predict dataset using fuzzy decision tree'
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
train_pred_path = args.outdir / f'fdt-train-pred-{output_name}.csv'
test_pred_path = args.outdir / f'fdt-test-pred-{output_name}.csv'
timing_path = args.outdir / f'fdt-timing-{output_name}.csv'

df_train_features = read_csv(train_features_path, index_col=0)
df_train_target = read_csv(train_target_path, index_col=0)
df_test_features = read_csv(test_features_path, index_col=0)

train_features = df_train_features.to_numpy()
train_target   = df_train_target.to_numpy()
test_features  = df_test_features.to_numpy()

train_index = df_train_features.index
test_index = df_test_features.index

model = Fuzzy_Decision_Tree_Regressor(
    min_split_size=int(configs['model']['min_split_size']),
    min_impurity_drop=float(configs['model']['min_impurity_drop']),
    max_num_splits=int(configs['model']['max_num_splits']),
    batch_size=int(configs['model']['batch_size']),
    n_epochs=int(configs['model']['n_epochs']),
)

fit_start = time()
model.fit(train_features, train_target)
fit_end = time()

train_pred_start = time()
train_pred = model.predict(train_features)
train_pred_end = time()

test_pred_start = time()
test_pred = model.predict(test_features)
test_pred_end = time()

df_train_pred = DataFrame(train_pred, columns=['pred'], index=train_index)
df_test_pred = DataFrame(test_pred, columns=['pred'], index=test_index)

df_train_pred.to_csv(train_pred_path)
df_test_pred.to_csv(test_pred_path)

fit_time = fit_end - fit_start
train_pred_time = train_pred_end - train_pred_start
test_pred_time = test_pred_end - test_pred_start

timing = DataFrame(
    [[fit_time, train_pred_time, test_pred_time]],
    columns=['fit', 'train_pred', 'test_pred'],
    index=['fdt']
)

timing.to_csv(timing_path)
