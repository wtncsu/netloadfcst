#!/usr/bin/env python3
from pandas import read_csv, merge
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

aparser = ArgumentParser(
    description='Plot point prediction results'
)

aparser.add_argument('--pred', type=Path, required=True)
aparser.add_argument('--target', type=Path, default=None)
args = aparser.parse_args()

pred = read_csv(args.pred, parse_dates=[0], index_col=0)

if args.target is None:
    plot_df = pred

else:
    target = read_csv(args.target, parse_dates=[0], index_col=0)
    plot_df = merge(target, pred, left_index=True, right_index=True)

title = f'pred={args.pred}\ntarget={args.target}'
plot_df.plot(marker='+', grid=True, title=title, ylabel='Outcome')
plt.show()
