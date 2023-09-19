#!/usr/bin/env python3
import tomllib
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument('--dataset_csv', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--settings', type=Path, required=True)
args = parser.parse_args()

with open(args.settings, 'rb') as settings_fd:
    settings = tomllib.load(settings_fd)

datetime_col = settings['Features']['datetime']
load_col = settings['Features']['load']
temperature_col = settings['Features']['temperature']

dataset = pd.read_csv(
    args.dataset_csv,
    usecols=[temperature_col, load_col, datetime_col],
    parse_dates=[datetime_col]
)

dataset['date'] = dataset[datetime_col].dt.date
dataset['hour'] = dataset[datetime_col].dt.hour

monthly_profiles = dict()
for month in range(1, 13):
    dataset_month = dataset[dataset[datetime_col].dt.month == month]
    profiles_month = dataset_month.pivot_table(
        index='date', columns='hour', values=load_col
    )
    profiles_month = profiles_month.bfill().ffill()
    monthly_profiles[month] = profiles_month

integer_to_month = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul',
    8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

args.output_dir.mkdir(exist_ok=True, parents=True)
for month, profile in monthly_profiles.items():
    output_filename = args.output_dir / (
        f'{args.dataset_csv.stem}-{integer_to_month[month]}.csv'
    )
    profile.to_csv(output_filename)
