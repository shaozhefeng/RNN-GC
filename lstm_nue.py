# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:04:54 2017

@author: LK
"""

from __future__ import print_function, division
import os

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU

from keras.callbacks import EarlyStopping


class Mylstm(object):
    def __init__(self, n_hidden=10, num_channel=5):
        if n_hidden is None:
            n_hidden = num_channel
        if num_channel == 1:  # keras could not accept drop when input_dim=1
            dropout_W = 0
            dropout_U = 0
        else:
            dropout_W = 0
            dropout_U = 0

        self.model = Sequential()
        #        self.model.add(LSTM(output_dim=n_hidden, input_dim=num_channel,
        #                            return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
        #        self.model.add(SimpleRNN(output_dim=n_hidden, input_dim=num_channel
        #                           #, init='glorot_normal'
        #                           #,  W_regularizer='l2', U_regularizer='l2'
        #                           #, dropout_W=0.2, dropout_U=0.2
        #                           ))


        self.model.add(LSTM(output_dim=n_hidden, input_dim=num_channel
                            # , init='glorot_normal'
                            # ,  W_regularizer='l2', U_regularizer='l2'
                            , dropout_W=dropout_W, dropout_U=dropout_U
                            # , inner_activation='linear' , activation = 'linear'
                            ))
        #        self.model.add(LSTM(output_dim=num_channel, input_dim=num_channel
        #               #, init='glorot_normal'
        #               #,  W_regularizer='l2', U_regularizer='l2'
        #               #, dropout_W=0.2, dropout_U=0.2
        #               #, inner_activation='linear' , activation = 'linear'
        #           ))
        #
        #
        ##        self.model.add(Dropout(0.2))
        #        self.model.add(LSTM(output_dim=n_hidden//2
        #                           #, init='glorot_normal'
        #                           #,  W_regularizer='l2', U_regularizer='l2'
        #                           # , activation = 'linear'
        #                           ))
        #
        #        self.model.add(Dropout(0.2))
        #        self.model.add(LSTM(output_dim=n_hidden,
        #                    return_sequences=False, dropout_W=0.2, dropout_U=0.2))
        #        self.model.add(Dropout(0.2))
        #        self.model.add(Dense(num_channel
        #                            # , init='glorot_normal'
        #                            # , W_regularizer='l2'
        #                            ))
        self.model.add(Dense(1
                             # , init='glorot_normal'
                             # , W_regularizer='l2'
                             ))
        # self.model.add(Activation('sigmoid'))

        self.model.summary()

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)  # lr=0.001, rho=0.9, epsilon=1e-6
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)  # lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        adagrad = Adagrad(lr=0.1, epsilon=1e-5)  # lr=0.01, epsilon=1e-5
        self.model.compile(loss='mean_squared_error'
                           , optimizer=rmsprop
                           )

    def fit(self, x, y, batch_size=10, nb_epoch=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        self.hist = self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split=0.2,
                                   callbacks=[early_stopping])
        #        self.hist = self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split =0.2)
        return self.hist

    def predict(self, x):
        #        return self.model.predict(x[:, np.newaxis, :])
        return self.model.predict(x)  ## main_nue


class Data_pre(object):
    @staticmethod
    def get_batch_sequence(x, seq_length=10, num_shift=1):
        # x: num_points * num_channel
        # return (samples, timesteps, input_dim)

        num_points = x.shape[0]
        inputs = []
        targets = []
        #        for p in np.arange(0, num_points, max(num_shift, seq_length // 5)):
        for p in np.arange(0, num_points, num_shift):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + num_shift >= num_points:
                break

            inputs.append(x[p: p + seq_length, :])
            # targets.append(x[p + num_shift: p + seq_length + num_shift, :])
            targets.append(x[p + seq_length, :])
        inputs = np.array(inputs)
        targets = np.array(targets)
        idx = np.random.permutation(np.arange(inputs.shape[0]))
        inputs = inputs[idx]
        targets = targets[idx]

        return inputs, targets
