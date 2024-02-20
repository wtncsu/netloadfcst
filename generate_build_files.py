#!/usr/bin/env python3

from glob import glob
from pathlib import Path

from ninja_syntax import Writer

writer = Writer(open('build.ninja', 'w'))


def generate(category):
    settings = [
        Path(file)
        for file in glob(f'settings/{category}/*.toml')
    ]

    for file in settings:
        config = f'{file}'
        train_feature = f'{category}_train/feature-{file.stem}.csv'
        train_target = f'{category}_train/target-{file.stem}.csv'
        test_feature = f'{category}_test/feature-{file.stem}.csv'
        test_target = f'{category}_test/target-{file.stem}.csv'
        predict_mean = f'output/{category}/mean-{file.stem}.csv'
        predict_std = f'output/{category}/std-{file.stem}.csv'
        visualize_tree = f'output/{category}/tree-{file.stem}.svg'
        train_time = f'output/{category}/train-time-{file.stem}'
        test_time = f'output/{category}/test-time-{file.stem}'
        plot = f'output/{category}/test-{file.stem}.png'

        build_command_name = f'{category}-{file.stem}'
        build_command = (
            './run_model.py '
            f'--config={config} '
            f'--train-feature={train_feature} '
            f'--train-target={train_target} '
            f'--test-feature={test_feature} '
            f'--predict-mean={predict_mean} '
            f'--predict-std={predict_std} '
            f'--visualize-tree={visualize_tree} '
            f'--train-time={train_time} '
            f'--test-time={test_time} '
        )

        plot_command_name = f'plot-{category}-{file.stem}'
        plot_command = (
            './plot_prediction.py '
            f'--mean={predict_mean} '
            f'--std={predict_std} '
            f'--target={test_target} '
            f'--save=$out '
        )

        writer.rule(name=build_command_name, command=build_command)
        writer.newline()

        writer.rule(name=plot_command_name, command=plot_command)
        writer.newline()

        writer.build(
            rule=build_command_name,
            outputs=[predict_mean, predict_std, visualize_tree],
            inputs=[train_feature, train_target, test_feature]
        )
        writer.default([predict_mean, predict_std, visualize_tree])
        writer.newline()

        writer.build(
            rule=plot_command_name,
            inputs=[test_target, predict_mean, predict_std],
            outputs=[plot]
        )

        writer.default(plot)
        writer.newline()


generate('netload_only')
generate('potential_only')
generate('netload')
generate('potential')
generate('combined')
