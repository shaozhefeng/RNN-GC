import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--sequence_length', type=int, default=20, help='sequence length')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--num_shift', type=int, default=1, help='the step size of the sequence')
        # self.parser.add_argument('--num_hidden', type=int, default=30, help='the number of hidden units')
        self.parser.add_argument('--num_epoch', type=int, default=100, help='the number of epoch')
        self.parser.add_argument('--theta', type=float, default=0.09, help='the first minimum error')
        self.parser.add_argument('--data_length', type=int, default=4096, help='length of data')
        self.parser.add_argument('--weight_decay', type=float, default=1e-7, help='regulization for the network ')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
