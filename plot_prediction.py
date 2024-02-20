#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num, DateFormatter
from pandas import read_csv
from scipy.stats import norm

parser = ArgumentParser()

parser.add_argument('--mean', type=Path, required=True)
parser.add_argument('--std', type=Path, required=True)
parser.add_argument('--target', type=Path)
parser.add_argument('--save', type=Path)
args = parser.parse_args()

mean = read_csv(args.mean, index_col=0)
std = read_csv(args.std, index_col=0)
target = (read_csv(args.target, index_col=0)
          if args.target is not None else None)

datetime = date2num(mean.index)


def annotate_plot(title):
    date_format = DateFormatter('%b %-d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)

    plt.xlabel('Time')
    plt.ylabel('Outcome')
    plt.xticks(rotation=15)
    plt.title(title)


def plot_heatmap(dist, target=None, q_low=1e-3, q_high=1 - 1e-3,
                 resolution=500):
    outcome_lo = dist.ppf(q_low).min()
    outcome_hi = dist.ppf(q_high).max()
    outcomes = np.linspace(outcome_lo, outcome_hi, resolution).reshape(-1, 1)
    density = dist.pdf(outcomes)

    plt.imshow(density, origin='lower', cmap='hot', aspect='auto',
               extent=(datetime.min(), datetime.max(), outcome_lo, outcome_hi))

    plt.grid(color='white', alpha=0.1)
    plt.colorbar(label='Prob. density')

    if target is not None:
        plt.scatter(datetime, target, color='white', s=1, label='Observations',
                    alpha=0.7)
        plt.legend()


def plot_interval(dist, target=None, q_low=0.05, q_high=0.95):
    band_lo = dist.ppf(q_low)
    band_hi = dist.ppf(q_high)

    if target is not None:
        plt.plot(datetime, target, marker='+', label='Observations')

    plt.plot(datetime, dist.mean(), label='Pred. mean')
    plt.fill_between(datetime, band_lo.ravel(), band_hi.ravel(), color='black',
                     alpha=0.2, label=f'Pred. band {q_low} ~ {q_high}')

    plt.legend()
    plt.grid()


for col in mean.columns:
    if target is not None:
        target_col = target[col]
    else:
        target_col = None

    title = (
        f'Pred mean={args.mean}\nstd={args.std}\n'
        f'Target={args.target}:{col}'
    )

    dist = norm(mean[col], std[col])

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(121)
    plot_heatmap(dist, target=target_col)
    annotate_plot(title)

    plt.subplot(122, sharex=ax1, sharey=ax1)
    plot_interval(dist, target=target_col)
    annotate_plot(title)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.19)

    if args.save is not None:
        filename = (
            f'{args.save.parent}/{args.save.stem}-{col}{args.save.suffix}'
        )

        plt.savefig(filename, dpi=300)

if args.save is None:
    plt.show()
