#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz


tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./rnn_log',
                        help='path to save log and checkpoint.')

    parser.add_argument('--text', type=str, default='QuanSongCi.txt',
                        help='path to QuanSongCi.txt')

    parser.add_argument('--num_steps', type=int, default=25,
                        help='number of time steps of one sample.')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size to use.')

    parser.add_argument('--dictionary', type=str, default='dictionary.npy',
                        help='path to dictionary.npy.')

    parser.add_argument('--reverse_dictionary', type=str, default='reverse_dictionary.npy',
                        help='path to reverse_dictionary.npy.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--embedding_file', type=str, default='embedding.npy',
                        help='lembedding_file')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    for x in dir(FLAGS):
        print(getattr(FLAGS, x))
