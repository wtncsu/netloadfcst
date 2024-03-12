#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from pandas import read_csv

parser = ArgumentParser()

parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--save', type=Path, required=True)
args = parser.parse_args()

dataset = read_csv(args.dataset, index_col='date', parse_dates=['date'])
dataset.clip(lower=0, inplace=True)
dataset.to_csv(args.save)
