#!/usr/bin/env python3
from pandas import read_csv, concat
from argparse import ArgumentParser
from pathlib import Path


aparser = ArgumentParser(
    description='Generate report'
)

aparser.add_argument('files', type=Path, action='extend', nargs='+')
aparser.add_argument('--out', type=Path, required=True)
args = aparser.parse_args()

dataframes = (
    read_csv(path, header=0)
    for path in args.files
)

df = concat(dataframes, axis=0, ignore_index=True)
df.index = (path.stem for path in args.files)
df.to_csv(args.out)
