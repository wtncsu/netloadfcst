#!/usr/bin/env python3
from glob import glob
from pathlib import Path
from types import SimpleNamespace


class Makefile:
    def __init__(self, filename='Makefile', default='all'):
        self._file = open(filename, 'w')
        self.print(f'.DEFAULT_GOAL:={default}')
        self.new_rule(default, depends=[])

    def close(self):
        self._file.close()

    def print(self, *args):
        print(*args, file=self._file)

    def phony(self, target):
        self.print(f'.PHONY: {target}')

    def new_rule(self, target_or_targets, depends, command=None):
        depends = ' '.join(depends)

        if isinstance(target_or_targets, list):
            target_or_targets = ' '.join(target_or_targets)
            self.print(f'{target_or_targets} &: {depends}')

        else:
            self.print(f'{target_or_targets} : {depends}')

        if command is not None:
            self.print(f'\t{command}')

        self.print()


def create_details(category, setting_file):
    details = SimpleNamespace()

    dataset = setting_file.stem

    details.setting = f'{setting_file}'
    details.train_feature = f'{category}_train/feature-{dataset}.csv'
    details.train_target = f'{category}_train/target-{dataset}.csv'
    details.test_feature = f'{category}_test/feature-{dataset}.csv'
    details.test_target = f'{category}_test/target-{dataset}.csv'

    details.predict_mean = f'output/{category}/mean-{dataset}.csv'
    details.predict_std = f'output/{category}/std-{dataset}.csv'
    details.train_time = f'output/{category}/train-time-{dataset}'
    details.test_time = f'output/{category}/test-time-{dataset}'

    details.visualize_tree = f'output/{category}/tree-{dataset}.svg'
    details.plot = f'output/{category}/{dataset}.png'

    details.dataset = dataset
    details.category = category
    return details


def run_model(details):
    run_target = f'run-{details.category}-{details.dataset}'

    targets = [
        f'{details.predict_mean}', f'{details.predict_std}',
        f'{details.train_time}', f'{details.test_time}',
        f'{details.visualize_tree}'
    ]

    depends = [
        f'{details.setting}', f'{details.train_feature}',
        f'{details.train_target}', f'{details.test_feature}'
    ]

    command = (
        './run_model.py '
        f'--config={details.setting} '
        f'--train-feature={details.train_feature} '
        f'--train-target={details.train_target} '
        f'--test-feature={details.test_feature} '
        f'--predict-mean={details.predict_mean} '
        f'--predict-std={details.predict_std} '
        f'--visualize-tree={details.visualize_tree} '
        f'--train-time={details.train_time} '
        f'--test-time={details.test_time} '
    )

    makefile.phony(run_target)
    makefile.new_rule(run_target, depends=targets)
    makefile.new_rule('all', depends=targets)
    makefile.new_rule(targets, depends=depends, command=command)


def plot_prediction(details):
    depends = [
        f'{details.predict_mean}', f'{details.predict_std}',
        f'{details.test_target}'
    ]

    makefile.new_rule(
        details.plot, depends=depends, command=(
            './plot_prediction.py '
            f'--mean={details.predict_mean} '
            f'--std={details.predict_std} '
            f'--target={details.test_target} '
            f'--save={details.plot} '
        )
    )

    makefile.new_rule('all', depends=[details.plot])


def show_prediction(details):
    show_target = f'show-{details.category}-{details.dataset}'

    depends = [
        f'{details.predict_mean}', f'{details.predict_std}',
        f'{details.test_target}'
    ]

    makefile.phony(show_target)
    makefile.new_rule(
        show_target, depends=depends, command=(
            './plot_prediction.py '
            f'--mean={details.predict_mean} '
            f'--std={details.predict_std} '
            f'--target={details.test_target} '
        )
    )


makefile = Makefile()

categories = [
    'netload', 'potential', 'potential_only', 'netload_only', 'combined'
]

for category in categories:
    setting_files = [
        Path(file) for file in glob(f'settings/{category}/*.toml')
    ]

    for setting_file in setting_files:
        details = create_details(category, setting_file)
        run_model(details)
        plot_prediction(details)
        show_prediction(details)

makefile.close()
