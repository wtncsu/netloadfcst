import torch
import argparse
import random
import numpy as np
from exp.main import Exp_Main


# Creating a new argument parser
parser = argparse.ArgumentParser(description='Training Transformer with pre-trained VAE-LSTM')

# Setting random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# Basic training configuration
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='Transformer_VAE_LSTM_S', help='model id')
parser.add_argument('--model', type=str, required=False, default='Transformer_VAE_LSTM',
                    help='model name, options: [Transformer_VAE_LSTM]')

parser.add_argument('--root_path', type=str,
                    default='../vae_lstm_transformer_multiple_singleoutput_S/vae_lstm/dataset_individual',
                    help='root path of the data file, options: [dataset_zone, dataset_individual]')
parser.add_argument('--data_path', type=str, default='DR_L1_MIDATL.csv', help='data file')
parser.add_argument('--data', type=str, required=False, default='DOEh', help='dataset type')
parser.add_argument('--saved_model_path', type=str,
                    default='../vae_lstm_transformer_multiple_singleoutput_S/vae_lstm/saved_models',
                    help='saved model file')
parser.add_argument('--vae_lstm_model', type=str, default='VAE_LSTM_demand_DOEh_DR_L1_MIDATL', help='vae lstm model')

# Setting model prediction mode, target and time encoding frequency -> Changed from S to MS due to bad results
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options: [M, S, MS]; '
                         'M:multivariate predict multivariate, S:univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='demand', help='target feature only used for S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'It can also be used for more detailed freq like 15min or 3h')

# Specifying forecasting task input and output lengths
parser.add_argument('--seq_len', type=int, default=24 * 14, help='input sequence length')
parser.add_argument('--label_len', type=int, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--interval_len', type=int, default=0, help='introduce interval in-between')

# Parameters for Transformers
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default '
                         '1: value embedding + temporal embedding + positional embedding '
                         '2: value embedding + temporal embedding '
                         '3: value embedding + positional embedding '
                         '4: value embedding')
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')   # Changed from 9

# Model dimensions
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
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
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')

# GPU + MPS settings
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_mps', type=bool, default=True, help='use mps')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Setting optimizer settings
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--momentum', type=float, default=0.1)

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
    'DR_L1_MIDATL.csv', 'DR_L1_SOUTH.csv', 'DR_L1_WEST.csv',
    'DR_L2_COAST.csv', 'DR_L2_EAST.csv', 'DR_L2_FWEST.csv', 'DR_L2_NCENT.csv', 'DR_L2_NORTH.csv', 'DR_L2_SCENT.csv',
    'DR_L2_SOUTH.csv', 'DR_L2_ZWEST.csv',
    'DR_L3_AT.csv', 'DR_L3_BE.csv', 'DR_L3_BG.csv', 'DR_L3_CH.csv', 'DR_L3_CZ.csv', 'DR_L3_DK.csv', 'DR_L3_ES.csv',
    'DR_L3_FR.csv', 'DR_L3_GR.csv', 'DR_L3_IT.csv', 'DR_L3_NL.csv', 'DR_L3_PT.csv', 'DR_L3_SI.csv', 'DR_L3_SK.csv',
    'DR_L4_CAPITL.csv', 'DR_L4_CENTRL.csv', 'DR_L4_DUNWOD.csv', 'DR_L4_GENESE.csv', 'DR_L4_HUD VL.csv',
    'DR_L4_LONGIL.csv', 'DR_L4_MILLWD.csv', 'DR_L4_N.Y.C..csv',
    'DR_L5_L5.csv'
]

Experiment = Exp_Main


# Looping over two individual forecasting tasks
for args.target in ['demand', 'potential']:

    if args.target == 'demand':
        vae_saved_model_list = [
            'VAE_LSTM_demand_DOEh_DR_L1_MIDATL', 'VAE_LSTM_demand_DOEh_DR_L1_SOUTH', 'VAE_LSTM_demand_DOEh_DR_L1_WEST',
            'VAE_LSTM_demand_DOEh_DR_L2_COAST', 'VAE_LSTM_demand_DOEh_DR_L2_EAST', 'VAE_LSTM_demand_DOEh_DR_L2_FWEST',
            'VAE_LSTM_demand_DOEh_DR_L2_NCENT', 'VAE_LSTM_demand_DOEh_DR_L2_NORTH', 'VAE_LSTM_demand_DOEh_DR_L2_SCENT',
            'VAE_LSTM_demand_DOEh_DR_L2_SOUTH', 'VAE_LSTM_demand_DOEh_DR_L2_ZWEST',
            'VAE_LSTM_demand_DOEh_DR_L3_AT', 'VAE_LSTM_demand_DOEh_DR_L3_BE', 'VAE_LSTM_demand_DOEh_DR_L3_BG',
            'VAE_LSTM_demand_DOEh_DR_L3_CH', 'VAE_LSTM_demand_DOEh_DR_L3_CZ', 'VAE_LSTM_demand_DOEh_DR_L3_DK',
            'VAE_LSTM_demand_DOEh_DR_L3_ES', 'VAE_LSTM_demand_DOEh_DR_L3_FR', 'VAE_LSTM_demand_DOEh_DR_L3_GR',
            'VAE_LSTM_demand_DOEh_DR_L3_IT', 'VAE_LSTM_demand_DOEh_DR_L3_NL', 'VAE_LSTM_demand_DOEh_DR_L3_PT',
            'VAE_LSTM_demand_DOEh_DR_L3_SI', 'VAE_LSTM_demand_DOEh_DR_L3_SK',
            'VAE_LSTM_demand_DOEh_DR_L4_CAPITL', 'VAE_LSTM_demand_DOEh_DR_L4_CENTRL',
            'VAE_LSTM_demand_DOEh_DR_L4_DUNWOD',
            'VAE_LSTM_demand_DOEh_DR_L4_GENESE', 'VAE_LSTM_demand_DOEh_DR_L4_HUD VL',
            'VAE_LSTM_demand_DOEh_DR_L4_LONGIL',
            'VAE_LSTM_demand_DOEh_DR_L4_MILLWD', 'VAE_LSTM_demand_DOEh_DR_L4_N',
            'VAE_LSTM_demand_DOEh_DR_L5_L5'
        ]
    elif args.target == 'potential':
        vae_saved_model_list = [
            'VAE_LSTM_potential_DOEh_DR_L1_MIDATL', 'VAE_LSTM_potential_DOEh_DR_L1_SOUTH',
            'VAE_LSTM_potential_DOEh_DR_L1_WEST',
            'VAE_LSTM_potential_DOEh_DR_L2_COAST', 'VAE_LSTM_potential_DOEh_DR_L2_EAST',
            'VAE_LSTM_potential_DOEh_DR_L2_FWEST',
            'VAE_LSTM_potential_DOEh_DR_L2_NCENT', 'VAE_LSTM_potential_DOEh_DR_L2_NORTH',
            'VAE_LSTM_potential_DOEh_DR_L2_SCENT',
            'VAE_LSTM_potential_DOEh_DR_L2_SOUTH', 'VAE_LSTM_potential_DOEh_DR_L2_ZWEST',
            'VAE_LSTM_potential_DOEh_DR_L3_AT',
            'VAE_LSTM_potential_DOEh_DR_L3_BE', 'VAE_LSTM_potential_DOEh_DR_L3_BG', 'VAE_LSTM_potential_DOEh_DR_L3_CH',
            'VAE_LSTM_potential_DOEh_DR_L3_CZ', 'VAE_LSTM_potential_DOEh_DR_L3_DK', 'VAE_LSTM_potential_DOEh_DR_L3_ES',
            'VAE_LSTM_potential_DOEh_DR_L3_FR', 'VAE_LSTM_potential_DOEh_DR_L3_GR', 'VAE_LSTM_potential_DOEh_DR_L3_IT',
            'VAE_LSTM_potential_DOEh_DR_L3_NL', 'VAE_LSTM_potential_DOEh_DR_L3_PT', 'VAE_LSTM_potential_DOEh_DR_L3_SI',
            'VAE_LSTM_potential_DOEh_DR_L3_SK',
            'VAE_LSTM_potential_DOEh_DR_L4_CAPITL', 'VAE_LSTM_potential_DOEh_DR_L4_CENTRL',
            'VAE_LSTM_potential_DOEh_DR_L4_DUNWOD',
            'VAE_LSTM_potential_DOEh_DR_L4_GENESE', 'VAE_LSTM_potential_DOEh_DR_L4_HUD VL',
            'VAE_LSTM_potential_DOEh_DR_L4_LONGIL',
            'VAE_LSTM_potential_DOEh_DR_L4_MILLWD', 'VAE_LSTM_potential_DOEh_DR_L4_N',
            'VAE_LSTM_potential_DOEh_DR_L5_L5',
        ]


    for args.data_path, args.vae_lstm_model in zip(dataset_list, vae_saved_model_list):

        data_name = args.data_path.split('.')[0]

        if args.is_training:
            # Setting record of experiment with hyperparameters identifier
            setting = '{}_{}_{}_{}_{}'.format(
                args.features,
                args.model_id,
                args.target,
                args.data,
                data_name)

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