# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import os

from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc import RNN_GC


def test(opt, num_hidden, mode, i):
    rnn_gc = RNN_GC(opt, num_hidden, mode)
    matrix = rnn_gc.nue()
    if not os.path.isdir('./inter_results'):
        os.makedirs('./inter_results')
    plot_save_intermediate_results(matrix, mode, i, './inter_results')
    return matrix


if __name__ == '__main__':
    num_test = 10
    opt = BaseOptions().parse()

    linear = np.zeros((5, 5))
    nonlinear = np.zeros((5, 5))
    nonlinear_lag = np.zeros((5, 5))
    for i in range(num_test):
        linear += test(opt, 30, 'linear', i)
        nonlinear += test(opt, 13, 'nonlinear', i)
        nonlinear_lag += test(opt, 30, 'nonlinearlag', i)

    linear /= num_test
    nonlinear /= num_test
    nonlinear_lag /= num_test
    plot_final_average_results(linear, nonlinear, nonlinear_lag, './', 1)
