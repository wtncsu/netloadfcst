#!/usr/bin/env python3
from pandas import read_csv
from configparser import ConfigParser
from argparse import ArgumentParser
from pathlib import Path

aparser = ArgumentParser(
    description='Preprocess datasets'
)

aparser.add_argument('config', type=Path)
aparser.add_argument('--datadir', type=Path, default='dataset')
aparser.add_argument('--outdir', type=Path, default='output')
args = aparser.parse_args()

args.outdir.mkdir(parents=True, exist_ok=True)

configs = ConfigParser()
configs.read(args.config)

dataset_file = args.datadir / f"{args.config.stem}.csv"
df = read_csv(dataset_file, index_col='time')

# add shifted columns
target_col = configs['dataset']['target_col']
for shift_name, shift_value in configs['shift'].items():
    shift_value = int(shift_value)
    df[shift_name] = df[target_col].shift(shift_value)

# this will drop empty rows created by shifting columns
df.dropna(inplace=True)


split_at = int(configs['split']['test_set'])
train = df[:split_at]
test  = df[split_at:]

train_target = train[target_col]
test_target  = test[target_col]
train_features = train.drop(columns=[target_col])
test_features  = test.drop(columns=[target_col])

output_name = args.config.stem
train_features_path = args.outdir / f'train-features-{output_name}.csv'
train_target_path = args.outdir / f'train-target-{output_name}.csv'
test_features_path = args.outdir / f'test-features-{output_name}.csv'
test_target_path = args.outdir / f'test-target-{output_name}.csv'
output_file_path = args.outdir / f'pp-{output_name}.done'
dependency_file_path = args.outdir / f'pp-{output_name}.dep'
depends_on = dataset_file

train_features.to_csv(train_features_path)
train_target.to_csv(train_target_path)
test_features.to_csv(test_features_path)
test_target.to_csv(test_target_path)

