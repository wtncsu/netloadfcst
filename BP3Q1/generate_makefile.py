#!/usr/bin/env python3
import sys
from glob import glob
from pathlib import Path

sys.stdout = open('Makefile', 'w')

dataset_paths = [Path(p) for p in glob('dataset/*.csv')]
datasets = [p.stem for p in dataset_paths]


def daily_profile_recipe(dataset):
    print(f'output/daily_profiles/{dataset}.done: \\')
    print(f'dataset/{dataset}.csv \\')
    print(f'settings/collect_daily_profiles/{dataset}.toml')
    print('\t./collect_daily_profiles.py \\')
    print(f'\t\t--dataset_csv dataset/{dataset}.csv \\')
    print('\t\t--output_dir output/daily_profiles \\')
    print(f'\t\t--settings settings/collect_daily_profiles/{dataset}.toml')
    print('\tdate > $@')
    print('')


def clustering_recipe(dataset, month):
    print(f'output/clusters/{dataset}-{month}.done: \\')
    print(f'output/daily_profiles/{dataset}.done \\')
    print(f'settings/generate_clusters/{dataset}.toml')
    print(f'\t./generate_clusters.py \\')
    print(f'\t\t--dataset_csv output/daily_profiles/{dataset}-{month}.csv \\')
    print(f'\t\t--output_dir output/clusters \\')
    print(f'\t\t--settings settings/generate_clusters/{dataset}.toml')
    print('\tdate > $@')
    print('')


def all_recipe(all_targets):
    all_targets_string = ' \\\n'.join(all_targets)
    print(f'all_targets:={all_targets_string}')
    print('')
    print('.PHONY: all')
    print('all: $(all_targets)')
    print('')


def clean_recipe():
    print('.PHONY: clean')
    print('clean:')
    print('\trm -rf output')
    print('')


def header():
    print('.DEFAULT_GOAL:=default')
    print('.PHONY: default')
    print('default:')
    print('\t./generate_makefile.py')
    print('\t$(MAKE) -C . all')
    print('')


months = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec'
]

all_targets = list()
for dataset in datasets:
    for month in months:
        all_targets.append(f'output/clusters/{dataset}-{month}.done')

header()
all_recipe(all_targets)
clean_recipe()

for dataset in datasets:
    daily_profile_recipe(dataset)

    for month in months:
        clustering_recipe(dataset, month)
