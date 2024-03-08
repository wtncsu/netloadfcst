#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

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

split_pos = {
    'L1-MIDATL':     34896,
    'L1-SOUTH':      34896,
    'L1-WEST':       34896,
    'L2-COAST':      34896,
    'L2-EAST':       34896,
    'L2-FWEST':      34896,
    'L2-NCENT':      34896,
    'L2-NORTH':      34896,
    'L2-SCENT':      34896,
    'L2-SOUTH':      34896,
    'L2-ZWEST':      34896,
    'L3-AT':         34896,
    'L3-BE':         34896,
    'L3-BG':         34896,
    'L3-CH':         34896,
    'L3-CZ':         34896,
    'L3-DK':         34896,
    'L3-ES':         34896,
    'L3-FR':         34896,
    'L3-GR':         34896,
    'L3-IT':         34896,
    'L3-NL':         34896,
    'L3-PT':         34896,
    'L3-SI':         34896,
    'L3-SK':         34896,
    'L4-CAPITAL':    34896,
    'L4-CENTRAL':    34896,
    'L4-DUNWOODY':   34896,
    'L4-GENESE':     34896,
    'L4-HUDSON':     34896,
    'L4-LONGISLAND': 34896,
    'L4-MILWOOD':    34896,
    'L4-NYC':        34896,
    'L5-DP':         34896,
    'S1':            -1465,
    'S2':            -1465,
    'S3-AT':         -8760,
    'S3-BE':         -8760,
    'S3-BG':         -8760,
    'S3-CH':         -8760,
    'S3-CZ':         -8760,
    'S3-DK':         -8760,
    'S3-ES':         -8760,
    'S3-FR':         -8760,
    'S3-GR':         -8760,
    'S3-IT':         -8760,
    'S3-NL':         -8760,
    'S3-PT':         -8760,
    'S3-SI':         -8760,
    'S3-SK':         -8760,
    'S4-MIDATL':     -4417,
    'S4-SOUTH':      -4417,
    'S4-WEST':       -4417,
    'S5':            -8760,
}

df_combined = [
    pd.read_csv(f'combined_datasets/{dataset}.csv', index_col=0)
    for dataset in datasets
]


# Columns: netload,mw_2,mw_7,potential,poten_2,poten_7

def case1(train_or_test):
    # mw_2,mw_7,poten_2,poten_7
    feature = train_or_test.drop(columns=['netload', 'potential'])
    target = train_or_test[['netload', 'potential']]
    return feature, target


def case2(train_or_test):
    # mw_2,mw_7,poten_2,poten_7
    feature = train_or_test.drop(columns=['netload', 'potential'])
    target = train_or_test[['potential']]
    return feature, target


def case3(train_or_test):
    # mw_2,mw_7
    feature = train_or_test.drop(
        columns=['netload', 'potential', 'poten_2', 'poten_7']
    )
    target = train_or_test[['netload']]
    return feature, target


def case4(train_or_test):
    # poten_2,poten_7
    feature = train_or_test.drop(
        columns=['netload', 'potential', 'mw_2', 'mw_7', 'kw_2', 'kw_7'],
        errors='ignore'
    )
    target = train_or_test[['potential']]
    return feature, target


def case5(train_or_test):
    # mw_2,mw_7,poten_2,poten_7
    feature = train_or_test.drop(columns=['netload', 'potential'])
    target = train_or_test[['netload']]
    return feature, target


cases = [
    case1, case2, case3, case4, case5
]

for df, dataset in zip(df_combined, datasets):
    pos = split_pos[dataset]

    train = df[:pos]
    test = df[pos:]

    print(f'writing dataset {dataset}')

    for case in cases:
        Path(f'{case.__name__}_train').mkdir(parents=True, exist_ok=True)
        Path(f'{case.__name__}_test').mkdir(parents=True, exist_ok=True)

        train_feature, train_target = case(train)
        test_feature, test_target = case(test)

        train_feature.to_csv(f'{case.__name__}_train/feature-{dataset}.csv')
        train_target.to_csv(f'{case.__name__}_train/target-{dataset}.csv')

        test_feature.to_csv(f'{case.__name__}_test/feature-{dataset}.csv')
        test_target.to_csv(f'{case.__name__}_test/target-{dataset}.csv')
