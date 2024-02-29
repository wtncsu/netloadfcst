#!/usr/bin/env python3
import pandas as pd

datasets = [
    'L1-MIDATL', 'L1-SOUTH', 'L1-WEST', 'L2-COAST', 'L2-EAST', 'L2-FWEST',
    'L2-NCENT', 'L2-NORTH', 'L2-SCENT', 'L2-SOUTH', 'L2-ZWEST', 'L3-AT',
    'L3-BE', 'L3-BG', 'L3-CH', 'L3-CZ', 'L3-DK', 'L3-ES', 'L3-FR', 'L3-GR',
    'L3-IT', 'L3-NL', 'L3-PT', 'L3-SI', 'L3-SK', 'L4-CAPITAL', 'L4-CENTRAL',
    'L4-DUNWOODY', 'L4-GENESE', 'L4-HUDSON', 'L4-LONGISLAND', 'L4-MILWOOD',
    'L4-NYC', 'L5-DP', 'S1', 'S2', 'S3-AT', 'S3-BE', 'S3-BG', 'S3-CH', 'S3-CZ',
    'S3-DK', 'S3-ES', 'S3-FR', 'S3-GR', 'S3-IT', 'S3-NL', 'S3-PT', 'S3-SI',
    'S3-SK', 'S4-MIDATL', 'S4-SOUTH', 'S4-WEST', 'S5'
]

df_potential = [
    pd.read_csv(f'potential_datasets/{dataset}.csv', parse_dates=[0],
                index_col=0)
    for dataset in datasets
]

df_netload = [
    pd.read_csv(f'netload_datasets/{dataset}.csv', parse_dates=[0], index_col=0)
    for dataset in datasets
]

for df in df_netload:
    df.index.name = 'date'
    df.rename(columns={'mw': 'netload', 'kw': 'netload'}, inplace=True)

for df in df_potential:
    df['poten_2'] = df['potential'].shift(48)
    df['poten_7'] = df['potential'].shift(168)

df_combined = list()

for df1, df2 in zip(df_netload, df_potential):
    duplicated1 = df1.index.duplicated(keep='first')
    duplicated2 = df2.index.duplicated(keep='first')

    dedup1 = df1[~duplicated1]
    dedup2 = df2[~duplicated2]

    combined = pd.concat((dedup1, dedup2), axis='columns')
    combined.dropna(axis='index', inplace=True)

    df_combined.append(combined)

for df, dataset in zip(df_combined, datasets):
    print(f'writing combined dataset {dataset}')
    df.to_csv(f'combined_datasets/{dataset}.csv')
