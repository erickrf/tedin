# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
An interactive shell for evaluating the numeric LSTM.
"""

import argparse
import tensorflow as tf
import numpy as np

import memorizer
from utils import Symbol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('load', help='File to load the model from')
    args = parser.parse_args()

    sess = tf.Session()
    model = memorizer.NumericLSTM(300, 9)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, args.load)
    num_time_steps = 9

    while True:
        sequence = raw_input('Type a sequence of numbers or X to exit: ')
        if sequence.upper() == 'X':
            break

        array = np.array([int(x) for x in sequence if x.isdigit()])
        sequence_size = len(array)
        if sequence_size > num_time_steps:
            print('Use at most %d digits', num_time_steps)
            continue

        padding_dims = (0, num_time_steps - len(array))
        padded = np.pad(array, padding_dims, 'constant', constant_values=Symbol.END)

        # we have to have the shape (time_steps, batch_size)
        transposed = np.array([padded]).T
        answer = model.run(sess, transposed, [sequence_size])[:, 0]
        answer_str = ' '.join(str(digit) for digit in answer)

        print('Model output: ', answer_str)

