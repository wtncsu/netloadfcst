#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()

parser.add_argument('dataset', type=Path, nargs='+')
parser.add_argument('--save', type=Path, required=True)
args = parser.parse_args()


def read_seconds(filename):
    with open(filename, 'r') as f:
        seconds = float(f.read())
    return seconds


filenames = [
    str(filename)
    for filename in args.dataset
]

times = [
    read_seconds(filename)
    for filename in args.dataset
]

report = pd.DataFrame({'filename': filenames, 'time': times})
report.set_index('filename', inplace=True)
report.sort_index(inplace=True)
report.to_csv(args.save)
