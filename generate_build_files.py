from glob import glob
from pathlib import Path

from ninja_syntax import Writer

writer = Writer(open('build.ninja', 'w'))

netload_only_settings = [
    Path(file)
    for file in glob('settings/netload_only/*.toml')
]

for file in netload_only_settings:
    config = f'{file}'
    train_feature = f'netload_only_train/feature-{file.stem}.csv'
    train_target = f'netload_only_train/target-{file.stem}.csv'
    test_feature = f'netload_only_test/feature-{file.stem}.csv'
    predict_mean = f'output/netload_only/mean-{file.stem}.csv'
    predict_std = f'output/netload_only/std-{file.stem}.csv'
    visualize_tree = f'output/netload_only/tree-{file.stem}.svg'
    train_time = f'output/netload_only/train-time-{file.stem}'
    test_time = f'output/netload_only/test-time-{file.stem}'

    rule_name = f'netload_only-{file.stem}'

    command = (
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

    writer.rule(name=rule_name, command=command)
    writer.newline()

    writer.build(
        rule=rule_name,
        outputs=[predict_mean, predict_std, visualize_tree],
        inputs=[train_feature, train_target, test_feature]
    )
    writer.newline()
