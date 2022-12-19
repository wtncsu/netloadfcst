#!/usr/bin/env python3
from pandas import read_csv, DataFrame, ExcelWriter
from argparse import ArgumentParser
from pathlib import Path

aparser = ArgumentParser(
    description='Convert result into Excel Spreadsheet'
)
aparser.add_argument('--outdir', type=Path, default='output')
aparser.add_argument('--model', type=str, required=True)
aparser.add_argument('filename', type=str)
args = aparser.parse_args()

test_metrics_file = args.outdir / f'{args.model}-test-metrics.csv'
train_metrics_file = args.outdir / f'{args.model}-train-metrics.csv'
timing_file = args.outdir / f'{args.model}-timing.csv'

test_metrics = read_csv(test_metrics_file, index_col=0)
train_metrics = read_csv(train_metrics_file, index_col=0)
timing = read_csv(timing_file, index_col=0)

timing_sheet = DataFrame(index=timing.index)
timing_sheet['fit (min)'] = timing['fit'] / 60
timing_sheet['train_pred (s)'] = timing['train_pred']
timing_sheet['test_pred (s)'] = timing['test_pred']

train_metrics_sheet = DataFrame(index=train_metrics.index)
train_metrics_sheet['peak load mape'] = train_metrics['mape_fs']
train_metrics_sheet.loc['average'] = train_metrics['mape_fs'].mean()

test_metrics_sheet = DataFrame(index=test_metrics.index)
test_metrics_sheet['peak load mape'] = test_metrics['mape_fs']
test_metrics_sheet.loc['average'] = test_metrics['mape_fs'].mean()

with ExcelWriter(args.filename) as workbook:
    test_metrics_sheet.to_excel(workbook, sheet_name='Test Metrics')
    timing_sheet.to_excel(workbook, sheet_name='Timing')
    train_metrics_sheet.to_excel(workbook, sheet_name='Train Metrics')
