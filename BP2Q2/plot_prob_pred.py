#!/usr/bin/env python3
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from argparse import ArgumentParser
from pathlib import Path
from scipy.stats import norm
import numpy as np

aparser = ArgumentParser(
    description='Plot probabilistic prediction results'
)

aparser.add_argument('--mean', type=Path, required=True)
aparser.add_argument('--std', type=Path, required=True)
aparser.add_argument('--target', type=Path, default=None)
args = aparser.parse_args()

mean = read_csv(args.mean, index_col=0)
std = read_csv(args.std, index_col=0)
target = (read_csv(args.target, index_col=0)
          if args.target is not None else None)
dist = norm(loc=mean, scale=std)

title = (
    f'Pred mean={args.mean}\nstd={args.std}\n'
    f'Target={args.target}'
)

band_prob_lo = 0.05
band_prob_hi = 0.95

band_lo = dist.ppf(band_prob_lo)
band_hi = dist.ppf(band_prob_hi)

if args.target is not None:
    outcome_lo = np.min((target, band_lo))
    outcome_hi = np.max((target, band_hi))
else:
    outcome_lo = target.min()
    outcome_hi = target.max()

resolution = 500
outcomes = np.linspace(outcome_lo, outcome_hi, resolution)
outcome_probs = dist.pdf(outcomes).transpose()
outcome_probs /= outcome_probs.max()
datetime = date2num(mean.index)
date_format = DateFormatter('%b %-d %H:%M')

figure = plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
extent = (datetime.min(), datetime.max(), outcome_lo, outcome_hi)
heatmap = plt.imshow(outcome_probs, extent=extent, aspect='auto',
                     origin='lower', cmap='hot')
ax1.xaxis.set_major_formatter(date_format)
plt.grid(color='white', alpha=0.1)
plt.colorbar(label='Probability density')

if target is not None:
    plt.scatter(datetime, target, color='white', marker='+', s=12,
                linewidths=0.5, label='Observations', alpha=0.9)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.xticks(rotation=45)
plt.title(title)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

if target is not None:
    plt.plot(datetime, target, marker='+', label='Observations')

plt.plot(datetime, mean, label='Prediction mean', marker='+')
plt.fill_between(datetime, band_lo.ravel(), band_hi.ravel(),
                 color='black', alpha=0.2,
                 label=f'Prediction band {band_prob_lo} ~ {band_prob_hi}')
ax2.xaxis.set_major_formatter(date_format)
plt.grid()
plt.legend()
plt.xlabel('Time')
plt.ylabel('Outcome')
plt.xticks(rotation=45)
plt.title(title)


plt.show()
