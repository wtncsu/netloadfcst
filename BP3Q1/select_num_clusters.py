#!/usr/bin/env python3
from glob import glob
import pandas as pd
from sklearn.metrics import silhouette_score

dataset_paths = glob('dataset/*.csv')
datasets = [
    path.replace('.csv', '').replace('dataset/', '')
    for path in dataset_paths
]

months = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
    'nov', 'dec'
]

clusters = range(3, 7)

scores = list()

for dataset in datasets:
    for month in months:

        for cluster in clusters:
            dataset_path = (f'output/clusters/{dataset}-{month}-'
                            f'{cluster}_clusters.csv')

            print(dataset_path)
            df = pd.read_csv(dataset_path, index_col='date')

            features = df.drop(columns='cluster_label').values
            labels = df['cluster_label'].values

            score = silhouette_score(features, labels)

            scores.append({
                'dataset': dataset,
                'month': month,
                'cluster': cluster,
                'score': score
            })

scores = pd.DataFrame.from_records(scores)
max_scores = scores.groupby(['dataset', 'month'])['score'].idxmax()
cluster_selection = scores.iloc[max_scores]
cluster_selection = cluster_selection.set_index(['dataset', 'month'], drop=True)
cluster_selection.to_csv('output/cluster_selection.csv')

alternative = cluster_selection.drop(columns=['score']).unstack('month')
alternative.to_csv('output/cluster_selection_alt.csv')

scores = scores.set_index(['dataset', 'month', 'cluster'], drop=True)
scores = scores.unstack('cluster')
scores.to_csv('output/silhouette_scores.csv')
