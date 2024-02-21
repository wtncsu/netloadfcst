#!/usr/bin/env python3
from glob import glob
from pathlib import Path
from types import SimpleNamespace


def _as_list(item_or_items):
    if isinstance(item_or_items, str):
        items = [item_or_items]

    else:
        items = list(item_or_items)

    return items


class Makefile:
    def __init__(self, file):
        self._file = file

    def print(self, *args):
        print(*args, file=self._file)

    def one_shell(self):
        self.print('.ONESHELL:')

    def set_default_goal(self, goals):
        goals = _as_list(goals)
        str_goals = ' '.join(goals)
        self.print(f'.DEFAULT_GOAL:={str_goals}')

    def add_default_goals(self, goals):
        goals = ' '.join(_as_list(goals))
        self.print(f'.DEFAULT_GOAL+={goals}')

    def add_rule(self, target, depends=None, command=None, phony=False,
                 grouped=False):

        target = ' '.join(_as_list(target))
        depends = '' if depends is None else ' '.join(_as_list(depends))
        command = None if command is None else (
            ' \n\t'.join(_as_list(command))
        )

        colon = ':' if not grouped else '&:'

        if phony:
            self.print(f'.PHONY: {target}')

        self.print(f'{target} {colon} {depends}')

        if command is not None:
            self.print(f'\t{command}')

        self.print()


def create_detail(setting_file):
    setting_folder = setting_file.parent.name
    dataset = setting_file.stem

    details = SimpleNamespace()

    details.dataset = dataset
    details.setting_folder = setting_folder

    details.setting_file = f'{setting_file}'
    details.train_feature = f'{setting_folder}_train/feature-{dataset}.csv'
    details.train_target = f'{setting_folder}_train/target-{dataset}.csv'
    details.test_feature = f'{setting_folder}_test/feature-{dataset}.csv'
    details.test_target = f'{setting_folder}_test/target-{dataset}.csv'

    details.predict = f'output/{setting_folder}/predict-{dataset}.csv'
    details.train_time = f'output/{setting_folder}/train-time-{dataset}'
    details.test_time = f'output/{setting_folder}/test-time-{dataset}'
    details.plot_predict = f'output/{setting_folder}/{dataset}.png'
    details.plot_tree = f'output/{setting_folder}/{dataset}.svg'

    return details


def run_model(detail, writer):
    targets = [
        f'{detail.predict}',
        f'{detail.train_time}',
        f'{detail.test_time}',
        f'{detail.plot_tree}'
    ]

    depends = [
        f'{detail.setting_file}',
        f'{detail.train_feature}',
        f'{detail.train_target}',
        f'{detail.test_feature}'
    ]

    command = (
        './run_model.py '
        f'--config={detail.setting_file} '
        f'--train-feature={detail.train_feature} '
        f'--train-target={detail.train_target} '
        f'--test-feature={detail.test_feature} '
        f'--predict={detail.predict} '
        f'--visualize-tree={detail.plot_tree} '
        f'--train-time={detail.train_time} '
        f'--test-time={detail.test_time} '
    )

    writer.add_rule(targets, depends=depends, command=command)

    writer.add_rule(f'run-{detail.setting_folder}-{detail.dataset}', phony=True,
                    depends=targets)

    writer.add_rule(f'run-{detail.setting_folder}', phony=True, depends=targets)
    writer.add_rule('all', phony=True, depends=targets)


def plot_prediction(detail, writer):
    depends = [
        f'{detail.predict}',
        f'{detail.test_target}'
    ]

    command = (
        './plot_prediction.py '
        f'--predict={detail.predict} '
        f'--target={detail.test_target} '
        f'--save={detail.plot_predict} '
    )

    target = detail.plot_predict
    phony_target1 = f'plot-{detail.setting_folder}-{detail.dataset}'
    phony_target2 = f'plot-{detail.setting_folder}'

    writer.add_rule(target, depends=depends, command=command)
    writer.add_rule(phony_target1, phony=True, depends=target)
    writer.add_rule(phony_target2, phony=True, depends=phony_target1)
    writer.add_rule('all', phony=True, depends=phony_target2)


def show_prediction(detail, writer):
    depends = [
        f'{detail.predict}',
        f'{detail.test_target}'
    ]

    command = (
        './plot_prediction.py '
        f'--predict={detail.predict} '
        f'--target={detail.test_target} '
        f'--save={detail.plot_predict} '
    )

    phony1 = f'show-{detail.setting_folder}-{detail.dataset}'
    phony2 = f'show-{detail.setting_folder}'

    writer.add_rule(phony1, phony=True, depends=depends, command=command)
    writer.add_rule(phony2, phony=True, depends=phony1)
    writer.add_rule('all', phony=True, depends=phony2)


def run_time_report(details, writer):
    save_time_report = 'output/time_report.csv'

    train_time_files = [detail.train_time for detail in details]
    test_time_files = [detail.test_time for detail in details]

    time_files = train_time_files + test_time_files
    str_time_files = ' '.join(time_files)

    command = f'./run_time_report.py --save {save_time_report} {str_time_files}'

    phony1 = 'time-report'

    writer.add_rule(save_time_report, depends=time_files, command=command)
    writer.add_rule(phony1, depends=save_time_report, phony=True)
    writer.add_rule('all', depends=phony1, phony=True)


def run_time_report_by_folder(details, writer):
    reports = dict()

    for detail in details:
        if detail.setting_folder not in reports:
            reports[detail.setting_folder] = list()

        reports[detail.setting_folder].append(detail.train_time)
        reports[detail.setting_folder].append(detail.test_time)

    for folder, time_files in reports.items():
        save_time_report = f'output/time_report-{folder}.csv'
        str_time_files = ' '.join(time_files)

        command = (
            './run_time_report.py '
            f'--save {save_time_report} '
            f'{str_time_files}'
        )

        phony1 = f'time-report-{folder}'

        writer.add_rule(save_time_report, depends=time_files, command=command)
        writer.add_rule(phony1, depends=save_time_report, phony=True)


def run_mape_report(details, writer):
    save_mape_report = 'output/mape_report.csv'

    predict_files = [detail.predict for detail in details]
    target_files = [detail.test_target for detail in details]

    files = predict_files + target_files
    str_files = ' '.join(files)

    command = list()
    command.append(f'./run_mape_report.py --save {save_mape_report}')

    for predict_file, target_file in zip(predict_files, target_files):
        command.append(f'--predict {predict_file} --target {target_file}')

    command = ' '.join(command)

    phony1 = 'mape-report'

    writer.add_rule(save_mape_report, depends=str_files, command=command)
    writer.add_rule(phony1, depends=save_mape_report, phony=True)
    writer.add_rule('all', depends=phony1, phony=True)


def run_mape_report_by_folder(details, writer):
    file_pairs = dict()

    for detail in details:
        if detail.setting_folder not in file_pairs:
            file_pairs[detail.setting_folder] = list()

        pair = (detail.predict, detail.test_target)
        file_pairs[detail.setting_folder].append(pair)

    for folder, pairs in file_pairs.items():
        save_mape_report = f'output/mape_report-{folder}.csv'

        command = list()
        command.append(f'./run_mape_report.py --save {save_mape_report}')

        for predict_file, target_file in pairs:
            command.append(f'--predict {predict_file} --target {target_file}')

        command = ' '.join(command)
        str_depends = ' '.join([
            file
            for pair in pairs
            for file in pair
        ])

        phony1 = f'mape-report-{folder}'

        writer.add_rule(save_mape_report, depends=str_depends, command=command)
        writer.add_rule(phony1, depends=save_mape_report, phony=True)


def generate_all():
    folders = [
        'netload', 'potential', 'potential_only', 'netload_only', 'combined'
    ]

    setting_files = [
        Path(file)

        for folder in folders
        for file in glob(f'settings/{folder}/*.toml')
    ]

    details = [
        create_detail(setting_file)
        for setting_file in setting_files
    ]

    with open('Makefile', 'w') as file:
        writer = Makefile(file)
        writer.one_shell()
        writer.set_default_goal('all')

        run_time_report(details, writer)
        run_time_report_by_folder(details, writer)

        run_mape_report(details, writer)
        run_mape_report_by_folder(details, writer)

        for detail in details:
            run_model(detail, writer=writer)
            plot_prediction(detail, writer=writer)
            show_prediction(detail, writer=writer)


generate_all()
