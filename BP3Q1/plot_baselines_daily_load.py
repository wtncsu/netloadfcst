from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

months = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec'
]

dataset_paths = glob('dataset/*.csv')
datasets = [
    path.replace('.csv', '').replace('dataset/', '')
    for path in dataset_paths
]

axes = plt.axes()

for dataset in datasets:
    for month in months:
        baseline_file = f'output/baselines/{dataset}-{month}.csv'
        daily_file = f'output/daily_profiles/{dataset}-{month}.csv'

        print(baseline_file)

        baseline = pd.read_csv(baseline_file, index_col='hour')
        daily = pd.read_csv(daily_file, index_col='date', parse_dates=['date'])

        plt.cla()
        daily.T.plot(ax=axes, legend=False, linewidth=1,
                     color='blue', alpha=0.3)
        baseline.plot(ax=axes, linewidth=1.5, color='red', y='load',
                      label=f'{month.title()} baseline')
        plt.grid()
        plt.title(f'Every Day and Baseline\n{dataset} {month.title()}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Load')

        plt.savefig(
            f'output/baselines/{dataset}-{month}.png',
            bbox_inches='tight', dpi=300
        )

