# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:33:44 2017

@author: LK
"""

from __future__ import print_function, division
import os

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
import scipy.io as sio
from lstm_nue import Mylstm, Data_pre
import copy
import datetime
import matplotlib.pyplot as plt


class RNN_GC(object):
    def __init__(self, seq_length, batch_size, num_shift, n_hidden, num_epoch, theta, data_length, option):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_shift = num_shift
        self.n_hidden = n_hidden
        self.num_epoch = num_epoch
        self.theta = theta
        self.data_length = data_length
        self.option = option

        save_dir = 'lstm_keras'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    def get_seq_data(self):
        data_2 = sio.loadmat(
            './simulation_difflen_' + self.option + '/realization_' + self.option + '_' + str(
                self.data_length) + '.mat')
        data = np.array(data_2["data"]).transpose()

        self.num_channel = data.shape[1]
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)  # scale data to [0. 1]

        x, y = Data_pre.get_batch_sequence(data, num_shift=self.num_shift,
                                           seq_length=self.seq_length)
        return x, y

    def lstm_nue(self):
        x, y = self.get_seq_data()

        granger_matrix = np.zeros([self.num_channel, self.num_channel])
        var_denominator = np.zeros([1, self.num_channel])
        all_candidate = []
        error_model = []
        error_all = []

        Hist_result = []
        starttime = datetime.datetime.now()

        for k in range(self.num_channel):

            tmp_y = np.reshape(y[:, k], [y.shape[0], 1])
            channel_set = range(self.num_channel)

            input_set = []
            last_error = 0

            for i in range(self.num_channel):

                min_error = 1e7
                min_idx = 0
                for x_idx in channel_set:
                    tmp_set = copy.copy(input_set)
                    tmp_set.append(x_idx)
                    tmp_x = x[:, :, tmp_set]

                    lstm = Mylstm(n_hidden=self.n_hidden, num_channel=len(tmp_set))
                    lstm.fit(tmp_x, tmp_y, batch_size=self.batch_size, nb_epoch=self.num_epoch)
                    tmp_error = np.mean((lstm.predict(tmp_x) - tmp_y) ** 2)
                    if tmp_error < min_error:
                        min_error = tmp_error
                        min_idx = x_idx
                    error_all.append([k, i, x_idx, tmp_error])
                error_model.append([k, last_error, min_error])
                if i != 0 and (np.abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                    break
                # print('the model of input number is %d' %len(tmp_set))
                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error

            all_candidate.append(input_set)
            lstm = Mylstm(n_hidden=self.n_hidden, num_channel=len(input_set))
            hist_res = lstm.fit(x[:, :, input_set], tmp_y, batch_size=self.batch_size, nb_epoch=self.num_epoch)
            Hist_result.append(hist_res)
            var_denominator[0][k] = np.var(lstm.predict(x[:, :, input_set]) - tmp_y, axis=0)
            for j in range(self.num_channel):
                if j not in input_set:
                    granger_matrix[j][k] = var_denominator[0][k]
                elif len(input_set) == 1:
                    tmp_x = x[:, :, k]
                    tmp_x = tmp_x[:, :, np.newaxis]
                    granger_matrix[j][k] = np.var(lstm.predict(tmp_x) - tmp_y, axis=0)
                else:
                    tmp_x = x[:, :, input_set]
                    channel_del_idx = input_set.index(j)
                    tmp_x[:, :, channel_del_idx] = 0
                    granger_matrix[j][k] = np.var(lstm.predict(tmp_x) - tmp_y, axis=0)

            print('train the model for %d output' % (k + 1))

        granger_matrix = granger_matrix / var_denominator
        for i in range(self.num_channel):
            granger_matrix[i][i] = 1
        granger_matrix[granger_matrix < 1] = 1
        granger_matrix = np.log(granger_matrix)
        plt.matshow(granger_matrix)
        plt.savefig('./lstm_keras/granger_' + str(self.data_length) + '_' + self.option + '.png')
        np.savetxt('./lstm_keras/granger_' + str(self.data_length) + '_' + self.option + '.txt', granger_matrix)
        # plt.show()

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).seconds
        print(interval)
        return granger_matrix


if __name__ == '__main__':
    seq_length = 20
    batch_size = 64
    num_shift = 1
    n_hidden = 15
    num_epoch = 1000
    theta = 0.08
    data_length = 4096

    rnn_gc_linear = RNN_GC(seq_length, batch_size, num_shift, n_hidden, num_epoch, theta, data_length, 'linear')
    linear = rnn_gc_linear.lstm_nue()

    rnn_gc_non = RNN_GC(seq_length, batch_size, num_shift, n_hidden, num_epoch, theta, data_length, 'nonlinear')
    nonlinear = rnn_gc_non.lstm_nue()

    rnn_gc_lag = RNN_GC(seq_length, batch_size, num_shift, n_hidden, num_epoch, theta, data_length, 'nonlinearlag')
    nonlinear_lag = rnn_gc_lag.lstm_nue()

    ground_truth = np.zeros((5, 5))
    ground_truth[0, 1] = 1
    ground_truth[0, 2] = 1
    ground_truth[0, 3] = 1
    ground_truth[3, 4] = 1
    ground_truth[4, 3] = 1

    plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(142)
    ax1.matshow(linear)
    ax1.axis('off')
    ax1.set_title('Linear')

    ax2 = plt.subplot(143)
    ax2.matshow(nonlinear)
    ax2.axis('off')
    ax2.set_title('Nonlinear')

    ax3 = plt.subplot(144)
    ax3.matshow(nonlinear_lag)
    ax3.axis('off')
    ax3.set_title('Nonlinear lag')

    ax4 = plt.subplot(141)
    ax4.matshow(ground_truth)
    ax4.axis('off')
    ax4.set_title('Ground Truth')

    plt.show()
    plt.savefig('all.png')
