import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
import time
import warnings
import numpy as np
import pandas as pd

from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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

    def vali(self, vali_data, vali_loader, criterion, is_test=True):
        total_loss = []
        total_true = []
        total_pred = []

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

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.model == 'CARD' and not is_test:

                    self.ratio = np.array([max(1 / np.sqrt(i + 1), 0.0) for i in range(self.args.pred_len)])
                    self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                    pred = outputs * self.ratio
                    true = batch_y * self.ratio
                    # loss = torch.mean(criterion(pred, true))
                else:
                    pred = outputs  # .detach().cpu()
                    true = batch_y  # .detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
                total_true.append(true)
                total_pred.append(pred)

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

        c = nn.L1Loss()
        # c = nn.MSELoss(reduction = 'none')
        # c = torch.nn.HuberLoss(reduction = 'none')

        self.warmup_epochs = self.args.warmup_epochs

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # Adding separate loss function for CARD
        c = nn.L1Loss()
        # c = nn.MSELoss(reduction = 'none')
        # c = torch.nn.HuberLoss(reduction = 'none')

        self.warmup_epochs = self.args.warmup_epochs

        def adjust_learning_rate_new(optimizer, epoch, args):
            """Decay the learning rate with half-cycle cosine after warmup"""
            min_lr = 0
            if epoch < self.warmup_epochs:
                lr = self.args.learning_rate * epoch / self.warmup_epochs
            else:
                lr = min_lr + (self.args.learning_rate - min_lr) * 0.5 * \
                     (1. + math.cos(
                         math.pi * (epoch - self.warmup_epochs) / (self.args.train_epochs - self.warmup_epochs)))

            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            print(f'Updating learning rate to {lr:.7f}')
            return lr

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            #             if self.args.lradj == 'CARD':
            #                 adjust_learning_rate_new(model_optim, epoch+1, self.args)
            #             else:
            #                 adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            # mae = 0
            # if epoch < self.warmup_epochs:
            #     mae = -1#0.1
            #     # for param_group in model_optim.param_groups:
            #     #     param_group['lr'] = 1e-3
            # else:
            #     mae = -1
            if self.args.lradj == 'CARD':
                adjust_learning_rate_new(model_optim, epoch + 1, self.args)
            elif self.args.lradj == 'constant':
                pass
            else:
                adjust_learning_rate(model_optim, epoch=epoch + 1, args=self.args, scheduler=scheduler)

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

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.model == 'CARD':
                        self.ratio = np.array([max(1 / np.sqrt(i + 1), 0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs * self.ratio
                        batch_y = batch_y * self.ratio

                    loss = c(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # if self.args.lradj == 'TST':
                #     adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                #     scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, c, is_test=False)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # if self.args.lradj != 'TST':
            #     adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            # else:
            #     print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

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
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr, mape_feat = metric(preds, trues)
        print('mape:{}, mse:{}, mae:{}, rse:{}'.format(mape, mse, mae, rse))
        print("feature-wise mape: {}".format(mape_feat))

        csv_folder_path = './csv_results/' + self.args.data_path.split('.')[0] + '/'
        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)

        # Saving the results in csv format
        import pandas as pd
        reshaped_mape_feat = pd.DataFrame(mape_feat)
        reshaped_mape_feat.to_csv(os.path.join(csv_folder_path, '{}_{}.csv'.format(self.args.target,
                                                                                   self.args.model)), index=False)

        # Saving results in text format
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mape:{}, mse:{}, mae:{}, rse:{}'.format(mape, mse, mae, rse))
        f.write("\nfeature-wise mape: {}".format(mape_feat))
        f.write('\n')
        f.write('\n')
        f.close()

        test_loss = np.average(test_loss)
        pred_df = pd.DataFrame(np.concatenate(preds))
        true_df = pd.DataFrame(np.concatenate(trues))

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # Creating a matplotlib plot
        for col_num in range(len(pred_df.columns)):
            comb_df = pd.concat([true_df.iloc[:, col_num],
                                 pred_df.iloc[:, col_num]], axis=1)
            ax = comb_df.plot(kind='line', title=setting)
            plt.savefig(plot_folder_path + str(col_num + 1) + '.png')

        # Saving best trained model
        torch.save(self.model, os.path.join(model_save_path + 'model_save.pth'))

        return