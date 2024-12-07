import argparse
import torch
# from exp.exp_main_optimized import Exp_Main
from exp.exp_main import Exp_Main
import random
import numpy as np


# Creating a new argument parser
parser = argparse.ArgumentParser(description='Transformer family for DOE day-ahead net-load plus DR forecasting')

# Setting random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# Basic training configuration
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='DOE_TEST_M', help='model id')  # Added MS
parser.add_argument('--model', type=str, required=False, default='CARD',
                    help='model name, options: [Autoformer, CARD, Informer, Linear, DLinear, NLinear, '
                         'PatchTST, Transformer]')

# Setting basic data path, checkpoint path and data name
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--root_path', type=str, default='./dataset_zone',
                    help='root path of the data file, options: [dataset_zone, dataset_individual]')
parser.add_argument('--data_path', type=str, default='DR_L1.csv', help='data file')
parser.add_argument('--data', type=str, required=False, default='DOEh', help='dataset type')

# Setting model prediction mode, target and time encoding frequency -> Changed from M to MS for task 2
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options: [M, S, MS]; '
                         'M:multivariate predict multivariate, S:univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='demand', help='target feature only used for S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'It can also be used for more detailed freq like 15min or 3h')

# Dataset specific args settings
parser.add_argument('--num_feats', type=int, default=3, help='number of features as input')
parser.add_argument('--data_col_len', type=int, default=24, help='total number of data columns')

# Specifying forecasting task input and output lengths
parser.add_argument('--seq_len', type=int, default=24 * 14, help='input sequence length')
parser.add_argument('--label_len', type=int, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--interval_len', type=int, default=0, help='introduce interval in-between')

# DLinear
# parser.add_argument('--individual', action='store_true', default=False, help='DLinear:
# a linear layer for each variate(channel) individually')

# Parameters for PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Parameters for Transformers
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default '
                         '1: value embedding + temporal embedding + positional embedding '
                         '2: value embedding + temporal embedding '
                         '3: value embedding + positional embedding '
                         '4: value embedding')

# DLinear with --individual, use enc_in hyperparameter as the number of channels
# Change this for different dataset -> Creating a lookup dictionary
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')   # Changed from 9

# Model dimensions
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')

# Model hyperparameters
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', default=True,
                    help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# Model optimization parameters
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU + MPS settings
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_mps', type=bool, default=True, help='use mps')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Setting optimizer settings
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--dp_rank', type=int, default=8)
parser.add_argument('--rescale', type=int, default=1)
parser.add_argument('--merge_size', type=int, default=2)
parser.add_argument('--momentum', type=float, default=0.1)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--devices_number', type=int, default=1)
parser.add_argument('--use_statistic', action='store_true', default=False)
parser.add_argument('--use_decomp', action='store_true', default=False)
parser.add_argument('--same_smoothing', action='store_true', default=False)
parser.add_argument('--warmup_epochs', type=int, default=0)

# Creating the final parser object
args = parser.parse_args()

# Setting the fixed seed for easy reproducibility
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Force to use GPU/MPS if available else fallback to CPU
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_mps = True if torch.backends.mps.is_available() and args.use_mps else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

models_list = [
    'Autoformer',
    'CARD',
    'Informer', 'Linear', 'DLinear', 'NLinear', 'PatchTST', 'Transformer'
]
dataset_list = [
    'DR_L1.csv',
    'DR_L2.csv',
    'DR_L3.csv',
    'DR_L4.csv',
    'DR_L5.csv'
]
dataset_len_list = [
    3 * args.num_feats,
    8 * args.num_feats,
    14 * args.num_feats,
    8 * args.num_feats,
    1 * args.num_feats
]

Experiment = Exp_Main

for args.data_path, args.data_col_len in zip(dataset_list, dataset_len_list):

    data_name = args.data_path.split('.')[0]
    args.enc_in = args.data_col_len
    args.dec_in = args.data_col_len
    args.c_out = args.data_col_len   # Comment out for MS task

    for args.model in models_list:

        print('Args in experiment:')
        print(args)

        # Adjust learning rate if using CARD
        if args.model == 'CARD':
            args.lradj = 'CARD'
            args.warmup_epochs = 10

        if args.is_training:
            for iter_time in range(args.itr):

                # Setting record of experiment with hyperparameters identifier
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_il{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    data_name,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.interval_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    iter_time)

                # Instantiating experiment with given args
                exp = Experiment(args)

                msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>Start training: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
                print(msg)
                exp.train(setting)

                msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>Testing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
                print(msg)
                exp.test(setting)

                # If prediction future unseen data is desired
                if args.do_predict:
                    msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>Predicting: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
                    print(msg)
                    exp.predict(setting, load=True)

                if args.use_gpu:
                    torch.cuda.empty_cache()
                elif args.use_mps:
                    torch.mps.empty_cache()
