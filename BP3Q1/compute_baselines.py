#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

output_dir = Path('output/baselines')
output_dir.mkdir(parents=True, exist_ok=True)

num_clusters = pd.read_csv('output/cluster_selection.csv')

for row in num_clusters.itertuples(index=False):
    dataset_path = (
        f'output/clusters/{row.dataset}-{row.month}-{row.cluster}_clusters.csv'
    )
    print(dataset_path)

    df = pd.read_csv(dataset_path, index_col='date')
    melted = df.melt(id_vars=['cluster_label'], var_name='hour',
                     value_name='load')
    cluster_means = melted.groupby('cluster_label')['load'].median()
    baseline_cluster = cluster_means.idxmin()

    baselines = df[df['cluster_label'] == baseline_cluster].copy()
    baselines.drop(inplace=True, columns=['cluster_label'])

    baseline = baselines.mean(axis=0)

    baseline = pd.DataFrame({'load': baseline})
    baseline.index.name = 'hour'
    baseline.to_csv(output_dir / f'{row.dataset}-{row.month}.csv')


