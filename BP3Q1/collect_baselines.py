#!/usr/bin/env python3
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

dataset_paths = glob('dataset/*.csv')
datasets = [
    path.replace('.csv', '').replace('dataset/', '')
    for path in dataset_paths
]

integer_to_month = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul',
    8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

axes = plt.axes()

for dataset in datasets:
    fragments = list()
    for month_val, month in integer_to_month.items():
        dataset_path = f'output/baselines/{dataset}-{month}.csv'
        print(dataset_path)
        df = pd.read_csv(dataset_path)

        df['month'] = month_val
        fragments.append(df)

    baseline = pd.concat(fragments, axis=0)
    baseline.set_index(['month', 'hour'], inplace=True)

    baseline.to_csv(f'output/baselines/{dataset}.csv')

    unstacked = baseline.unstack('month').droplevel(0, axis=1)

    plt.cla()
    unstacked.plot(ax=axes)
    plt.title(f'Baseline Daily Profiles\n{dataset}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load')
    plt.grid()
    plt.savefig(f'output/baselines/{dataset}.png',
                bbox_inches='tight', dpi=300)

