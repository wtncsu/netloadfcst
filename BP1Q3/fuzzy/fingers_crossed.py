#!/usr/bin/env python3
import pandas
import numpy
from datools.regression.fuzzy_decision_trees import (
    Fuzzy_Decision_Tree_Regressor,
)

from datools.metrics.regression import (
    mean_absolute_percent_error as mape,
    mean_absolute_percent_full_scale_error as mapefs,
    weighted_mean_absolute_percent_error as wmape,
)

from datools.gradients.optimizers import (
    Adam,
)

from matplotlib import pyplot as plt

from configparser import ConfigParser
from argparse import ArgumentParser

aparser = ArgumentParser(
    description='Fit fuzzy decision tree'
)

aparser.add_argument('--csv', type=str, required=True)
aparser.add_argument('--config', type=str, required=True)
aparser.add_argument('--output', type=str, required=True)
args = aparser.parse_args()

config = ConfigParser()
with open(args.config) as config_file:
    config.read_file(config_file)

target_col = config['data']['target']
split_at = int(config['data']['train_test_split'])
min_count = int(config['architecture']['min_count'])
min_impurity_drop = int(config['architecture']['min_impurity_drop'])
batch_size = int(config['tune']['batch_size'])
epochs = int(config['tune']['epochs'])

df = pandas.read_csv(args.csv)
for shift_name, shift_value in config['shift'].items():
    shift_value = int(shift_value)
    df[shift_name] = df[target_col].shift(shift_value)

df.dropna(inplace=True)

train = df[:split_at].reset_index(drop=True)
test  = df[split_at:].reset_index(drop=True)

y_train = train[target_col]
y_test  = test[target_col]
x_train = train.drop(columns=[target_col])
x_test  = test.drop(columns=[target_col])

model = Fuzzy_Decision_Tree_Regressor(
    min_impurity_drop=min_impurity_drop,
    min_count=min_count)

model.fit(x_train, y_train)
yhat_crisp_train = model.predict(x_train)
yhat_crisp_test = model.predict(x_test)

loss = model.tune(
    x_train, y_train, batch_size=batch_size, epochs=epochs,
    ybar_optimizer=Adam(), gain_optimizer=Adam(),
    threshold_optimizer=Adam()
)
yhat_tune_train = model.predict(x_train)
yhat_tune_test = model.predict(x_test)

df_loss = pandas.DataFrame({
    'mean': loss.mean(axis=1),
    '90%': numpy.quantile(loss, 0.9, axis=1),
    '10%': numpy.quantile(loss, 0.1, axis=1)
}).plot(title='Loss')
plt.savefig(f'{args.output}/loss.png')


df_train = pandas.DataFrame({
    'y': y_train,
    'yhat_crisp': yhat_crisp_train,
    'yhat_tune': yhat_tune_train
})
df_train.to_csv(f'{args.output}/train.csv')
df_train.plot(title='Train')
plt.savefig(f'{args.output}/train.png')


df_test = pandas.DataFrame({
    'y': y_test,
    'yhat_crisp': yhat_crisp_test,
    'yhat_tune': yhat_tune_test
})
df_test.to_csv(f'{args.output}/test.csv')
df_test.plot(title='Test')
plt.savefig(f'{args.output}/test.png')

metrics = (
    mape,
    mapefs,
    wmape,
)

pairs = {
    'train': ({
        'crisp': yhat_crisp_train,
        'tune' : yhat_tune_train,
    }, y_train),

    'test': ({
        'crisp' : yhat_crisp_test,
        'tune'  : yhat_tune_test,
    }, y_test)
}

with open(f'{args.output}/result', 'w') as result:
    for train_or_test, (series, actual) in pairs.items():
            for metric in metrics:
                crisp_metric = metric(series['crisp'], actual)
                tune_metric = metric(series['tune'], actual)
                improved = (crisp_metric - tune_metric) / crisp_metric * 100

                header = f'{train_or_test}-{metric.__name__}'
                result.write(f'{header}-crisp={crisp_metric}\n')
                result.write(f'{header}-tune={tune_metric}\n')
                result.write(f'{header}-improved={improved}\n')

