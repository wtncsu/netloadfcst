#!/usr/bin/env python3
import tomllib
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

output_path = Path('output/technical_potential')
output_path.mkdir(parents=True, exist_ok=True)

dataset_paths = glob('dataset/*.csv')
datasets = [
    path.replace('.csv', '').replace('dataset/', '')
    for path in dataset_paths
]

axes = plt.axes()

for dataset in datasets:
    dataset_file = f'dataset/{dataset}.csv'
    settings_file = f'settings/collect_daily_profiles/{dataset}.toml'
    baseline_file = f'output/baselines/{dataset}.csv'

    print(dataset_file)

    with open(settings_file, 'rb') as settings_fd:
        settings = tomllib.load(settings_fd)

    datetime_col = settings['Features']['datetime']
    load_col = settings['Features']['load']
    temperature_col = settings['Features']['temperature']

    df = pd.read_csv(dataset_file,
                     usecols=[load_col, datetime_col],
                     parse_dates=[datetime_col], index_col=datetime_col)
    df.index.name = 'date'
    df.columns = ['load']
    df['month'] = df.index.month
    df['hour'] = df.index.hour

    baseline = pd.read_csv(baseline_file, index_col=['month', 'hour'])
    baseline.rename(columns={'load': 'baseline'}, inplace=True)
    merged = df.reset_index().merge(baseline, how='left', on=['month', 'hour'])
    merged.set_index('date', drop=True, inplace=True)

    merged['potential'] = merged['load'] - merged['baseline']

    eligible_load = merged[['potential']].clip(lower=0)
    eligible_load.to_csv(output_path / f'{dataset}.csv')

    plt.cla()
    eligible_load.plot(ax=axes)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Technical Potential')
    plt.title(f'Technical DR Potential\n{dataset}')
    plt.savefig(output_path / f'{dataset}.png',
                bbox_inches='tight', dpi=300)
    
    plt.cla()
    eligible_load.plot.hist(ax=axes)
    plt.grid()
    plt.xlabel('Technical Potential')
    plt.title(f'Technical DR Potential Histogram\n{dataset}')
    plt.savefig(output_path / f'{dataset}_hist.png',
                bbox_inches='tight', dpi=300)

