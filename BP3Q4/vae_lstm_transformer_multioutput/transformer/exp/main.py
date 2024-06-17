import os
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from utils.tools import EarlyStopping
from utils.metrics import metric
from data_provider.data_factory import data_provider
from model import vae_lstm, vanilla_transformer, vae_transformer
from exp.basic import Exp_Basic
import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        vae_model = torch.load(os.path.join(self.args.saved_model_path,
                                            self.args.vae_lstm_model, 'model_save.pth'))
        transformer_model = vanilla_transformer.Model(self.args).float()

        model = vae_transformer.VAE_Transformer(vae_model, transformer_model)

        # Individual models
        # model = vanilla_transformer.Model(self.args).float()
        # model = torch.load(os.path.join(self.args.saved_model_path,
        #                                     self.args.vae_lstm_model, 'model_save.pth'))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def init_weights(model):
        if isinstance(model, nn.Linear):
            torch.nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
            model.bias.data.fill_(0.01)

    def load_from_checkpoint(model, filename):
        model.load_state_dict(torch.load(filename))
        return model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                # Continuous features
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # Calendar features
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # out_var, out_transformer = self.model(batch_x, batch_x_mark)
                # m_loss, out_transformer, info = self.model(batch_x)

                # out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out_var, out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Reshaping outputs
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = out_transformer[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs
                true = batch_y

                # Defining loss function
                loss = criterion(pred, true)

                # Resetting the calculated gradients
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_results_folder_path = './test_result_timing/'
        if not os.path.exists(time_results_folder_path):
            os.makedirs(time_results_folder_path)

        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()
        time_train_s = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # Continuous features
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Calendar features
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # out_var, out_transformer = self.model(batch_x, batch_x_mark)
                # m_loss, out_transformer, info = self.model(batch_x)

                # out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out_var, out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Reshaping outputs
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = out_transformer[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Defining loss function
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # Taking the optimization step
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = checkpoint_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        training_time = time.time() - time_train_s
        training_time_df = pd.DataFrame({'time': [training_time]})
        training_time_df.to_csv(time_results_folder_path + '{}.csv'.format(setting), index=False)

        return self.model

    def test(self, setting, test=0):
        import pandas as pd
        test_data, test_loader = self._get_data(flag='test')
        test_loss = []

        plot_folder_path = './test_results_plot/' + setting + '/'
        if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

        model_save_path = './saved_models/' + setting + '/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                # Continuous features
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Calendar features
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # out_var, out_transformer = self.model(batch_x, batch_x_mark)
                # m_loss, out_transformer, info = self.model(batch_x)

                # out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                out_var, out_transformer = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # print("INPUT SHAPE: ", batch_x.shape)
                # print("PRED SHAPE: ", out_transformer.shape)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = out_transformer[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr, mape_feat = metric(preds, trues)
        print('mape:{}, mse:{}, mae:{}, rse:{}'.format(mape, mse, mae, rse))
        print("feature-wise mape: {}".format(mape_feat))

        csv_folder_path = './csv_results/' + self.args.data_path.split('.')[0] + '/'
        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)

        num_of_cols = self.args.data_col_len // self.args.num_feats

        # Saving the results in csv format
        reshaped_mape_feat = pd.DataFrame(mape_feat.reshape(num_of_cols, -1))
        reshaped_mape_feat.to_csv(os.path.join(csv_folder_path, '{}.csv'.format(self.args.model)), index=False)

        test_loss = np.average(test_loss)
        pred_df = pd.DataFrame(np.concatenate(preds))
        true_df = pd.DataFrame(np.concatenate(trues))

        # Creating a matplotlib plot
        for col_num in range(len(pred_df.columns)):
            comb_df = pd.concat([true_df.iloc[:, col_num],
                                 pred_df.iloc[:, col_num]], axis=1)
            ax = comb_df.plot(kind='line', title=setting)
            plt.savefig(plot_folder_path + str(col_num+1) + '.png')

        # Saving best trained model
        torch.save(self.model, os.path.join(model_save_path + 'model_save.pth'))

        return