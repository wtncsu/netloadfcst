#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


aparser = ArgumentParser(
    description='Combine probabilistic forecast'
)

aparser.add_argument('--mean', type=Path, required=True)
aparser.add_argument('--std', type=Path, required=True)
aparser.add_argument('--out', type=Path, required=True)
args = aparser.parse_args()

mean = pd.read_csv(args.mean, index_col=0, parse_dates=[0])
std = pd.read_csv(args.std, index_col=0, parse_dates=[0])

combined = pd.concat((mean, std), axis=1)
combined.to_csv(args.out)
