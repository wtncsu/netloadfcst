#!/usr/bin/env python3

import tomllib
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans

parser = ArgumentParser()
parser.add_argument('--dataset_csv', type=Path, required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--settings', type=Path, required=True)
args = parser.parse_args()

with open(args.settings, 'rb') as settings_fd:
    settings = tomllib.load(settings_fd)

min_clusters = settings['Clustering']['n_clusters_min']
max_clusters = settings['Clustering']['n_clusters_max']

profiles = pd.read_csv(args.dataset_csv, index_col='date')
features = profiles.values

num_clusters_list = range(min_clusters, max_clusters + 1)
labels_list = list()
score_list = list()

args.output_dir.mkdir(parents=True, exist_ok=True)

for n_clusters in (progress := tqdm(num_clusters_list)):
    progress.set_postfix_str(f'KMeans {n_clusters=:}', refresh=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw')
    labels = kmeans.fit_predict(scaled).astype(int)

    score = silhouette_score(features, labels)

    labels_list.append(labels)
    score_list.append(score)

for num_clusters, labels in zip(num_clusters_list, labels_list):
    output_file_path = args.output_dir / (
        f'{args.dataset_csv.stem}-{num_clusters}_clusters.csv'
    )

    output_dataset = profiles.copy()
    output_dataset['cluster_label'] = labels

    output_dataset.to_csv(output_file_path)

clusters_list = [
    [profiles[labels == cluster_id] for cluster_id in range(num_clusters)]
    for num_clusters, labels in zip(num_clusters_list, labels_list)
]

cluster_colormap = matplotlib.colormaps['Dark2']

for num_clusters, clusters in zip(num_clusters_list, clusters_list):
    plt.figure()
    cluster_plot_axes = plt.axes()
    for cluster_id, cluster in enumerate(clusters):
        cluster.T.plot(
            ax=cluster_plot_axes, linewidth=1.5, legend=False,
            color=cluster_colormap(cluster_id),
            alpha=0.5
        )

    plt.title(f'Daily Profiles ({num_clusters=:})\n{args.dataset_csv.stem}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load')
    plt.grid()
    plt.savefig(
        args.output_dir /
        f'{args.dataset_csv.stem}-{num_clusters}_clusters.png',
        bbox_inches='tight', dpi=300
    )

plt.figure()
plt.bar(num_clusters_list, score_list)
plt.xlabel('Number of KMeans Clusters')
plt.ylabel('Silhouette Score')
plt.title(f'Silhouette Scores \n{args.dataset_csv.stem}')
plt.grid()
plt.savefig(args.output_dir / f'{args.dataset_csv.stem}.png',
            bbox_inches='tight', dpi=300)
