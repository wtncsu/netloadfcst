import os
import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from utils.tools import EarlyStopping
from data_provider.data_factory import data_provider
from model.vae_lstm import VAE_LSTM
from exp.basic import Exp_Basic
import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].VAE_LSTM(self.args).float()

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

                m_loss, recon_x, info = self.model(batch_x)

                # Resetting the calculated gradients
                total_loss.append(m_loss.mean().item())

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
                batch_y = batch_y.float()

                # Calendar features
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                m_loss, recon_x, info = self.model(batch_x)
                train_loss.append(m_loss.mean().item())

                # Taking the optimization step
                m_loss.mean().backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, m_loss.mean().item()))
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
                batch_y = batch_y.float()

                # Calendar features
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                m_loss, recon_x, info = self.model(batch_x)

                preds.append(recon_x.cpu().detach().numpy().reshape(-1, self.args.output_size))
                trues.append(batch_x.cpu().detach().numpy().reshape(-1, self.args.output_size))
                test_loss.append(m_loss.mean().item())

        test_loss = np.average(test_loss)
        pred_df = pd.DataFrame(np.concatenate(preds))
        true_df = pd.DataFrame(np.concatenate(trues))

        print("Average test loss: ", test_loss)

        # Creating a matplotlib plot
        for col_num in range(len(pred_df.columns)):
            comb_df = pd.concat([true_df.iloc[:, col_num],
                                 pred_df.iloc[:, col_num]], axis=1)
            ax = comb_df.plot(kind='line', title=setting)
            plt.savefig(plot_folder_path + str(col_num+1) + '.png')

        # Saving best trained model
        torch.save(self.model, os.path.join(model_save_path + 'model_save.pth'))

        return
