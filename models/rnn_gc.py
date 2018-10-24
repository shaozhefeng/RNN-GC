# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import copy
import datetime

import numpy as np
from sklearn import preprocessing
import scipy.io as sio

from models.custom_lstm import CustomLSTM
from util.util import batch_sequence


class RNN_GC(object):
    def __init__(self, opt, num_hidden, mode):
        self.sequence_length = opt.sequence_length
        self.batch_size = opt.batch_size
        self.num_shift = opt.num_shift
        self.num_hidden = num_hidden
        self.num_epoch = opt.num_epoch
        self.theta = opt.theta
        self.data_length = opt.data_length
        self.weight_decay = opt.weight_decay

        self.mode = mode

    def load_sequence_data(self):
        simulation_name = 'realization_' + self.mode + '_' + str(self.data_length) + '.mat'
        simulation_dir = 'simulation_difflen'
        simulation_name = os.path.join(simulation_dir, simulation_name)
        simulation_data = sio.loadmat(simulation_name)

        simulation_data = np.array(simulation_data["data"]).transpose()

        self.num_channel = simulation_data.shape[1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        simulation_data = scaler.transform(simulation_data)
        min_max_scaler = preprocessing.MinMaxScaler()

        # scale data to [0. 1]
        data = min_max_scaler.fit_transform(simulation_data)

        x, y = batch_sequence(data, num_shift=self.num_shift,
                              sequence_length=self.sequence_length)
        return x, y

    def nue(self):
        x, y = self.load_sequence_data()

        granger_matrix = np.zeros([self.num_channel, self.num_channel])
        var_denominator = np.zeros([1, self.num_channel])
        all_candidate = []
        error_model = []
        error_all = []

        hist_result = []
        start_time = datetime.datetime.now()

        for k in range(self.num_channel):

            tmp_y = np.reshape(y[:, k], [y.shape[0], 1])
            channel_set = list(range(self.num_channel))

            input_set = []
            last_error = 0

            for i in range(self.num_channel):

                min_error = 1e7
                min_idx = 0
                for x_idx in channel_set:
                    tmp_set = copy.copy(input_set)
                    tmp_set.append(x_idx)
                    tmp_x = x[:, :, tmp_set]

                    lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(tmp_set),
                                      weight_decay=self.weight_decay)
                    lstm.fit(tmp_x, tmp_y, batch_size=self.batch_size, epochs=self.num_epoch)
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
            lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
            hist_res = lstm.fit(x[:, :, input_set], tmp_y, batch_size=self.batch_size, epochs=self.num_epoch)
            hist_result.append(hist_res)
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

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds
        print('training time: %d seconds' % int(interval))
        return granger_matrix
