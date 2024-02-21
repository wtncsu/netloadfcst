#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from pandas import read_csv

parser = ArgumentParser()

parser.add_argument('--predict', type=Path, required=True)
parser.add_argument('--target', type=Path)
parser.add_argument('--save', type=Path)
args = parser.parse_args()


def parse_datasets():
    predict = read_csv(args.predict, index_col=0, parse_dates=[0])
    target = None
    if args.target is not None:
        target = read_csv(args.target, index_col=0, parse_dates=[0])

    return predict, target


def plot(predict, target):
    plt.figure(figsize=(12, 4))

    if target is not None:
        plt.plot(target, label='target', marker='+')

    plt.plot(predict, label='predict', marker='+')

    plt.xlabel('Time')
    plt.ylabel('Outcome')
    plt.xticks(rotation=15)
    plt.grid()
    plt.legend()


def plot_all():
    predict, target = parse_datasets()

    for column in predict.columns:
        plot(
            predict=predict[column],
            target=(None if target is None else target[column])
        )

        plt.title(f'predict={args.predict}:{column}')

        if args.save is not None:
            plt.savefig(f'{args.save}/{column}.png', dpi=300)

    if args.save is None:
        plt.show()


plot_all()
