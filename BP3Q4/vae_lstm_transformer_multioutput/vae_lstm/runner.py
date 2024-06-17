import torch
import argparse
import random
import numpy as np
from exp.main import Exp_Main


# Creating a new argument parser
parser = argparse.ArgumentParser(description='Pre-training VAE-LSTM')

# Setting random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# Basic training configuration
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='VAE_LSTM', help='model id')
parser.add_argument('--model', type=str, required=False, default='VAE_LSTM',
                    help='model name, options: [VAE_LSTM]')

parser.add_argument('--root_path', type=str, default='./dataset_zone',
                    help='root path of the data file, options: [dataset_zone, dataset_individual]')
parser.add_argument('--data_path', type=str, default='DR_L1.csv', help='data file')
parser.add_argument('--data', type=str, required=False, default='DOEh', help='dataset type')

parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='output sequence length')

parser.add_argument('--input_size', type=int, default=3, help='input size')
parser.add_argument('--hidden_size', type=int, default=2048, help='hidden layer dim')
parser.add_argument('--latent_size', type=int, default=1024, help='latent space dim')   # Changed from 9
parser.add_argument('--output_size', type=int, default=3, help='output size')

parser.add_argument('--num_feats', type=int, default=3, help='number of features as input')
parser.add_argument('--data_col_len', type=int, default=24, help='total number of data columns')

parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'It can also be used for more detailed freq like 15min or 3h')

# GPU + MPS settings
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_mps', type=bool, default=True, help='use mps')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

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
    args.input_size = args.data_col_len
    args.output_size = args.data_col_len

    if args.is_training:
        # Setting record of experiment with hyperparameters identifier
        setting = '{}_{}_{}_{}'.format(
            args.model_id,
            args.data,
            data_name,
            args.input_size)

        # Instantiating experiment with given args
        exp = Experiment(args)

        msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>Start training: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
        print(msg)
        exp.train(setting)

        msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>Testing: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
        print(msg)
        exp.test(setting)

        if args.use_gpu:
            torch.cuda.empty_cache()
        elif args.use_mps:
            torch.mps.empty_cache()