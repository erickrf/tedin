# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
An interactive shell for evaluating the memorizer auto-encoder.
"""

import argparse
import tensorflow as tf
import numpy as np

import memorizer
import addition
import utils
from utils import Symbol


def _read_sequence_from_prompt(max_size):
    """
    Read a sequence from the input prompt and enforce its maximum size.
    :return: the sequence or None (exit command)
    """
    ready = False
    while not ready:
        sequence = raw_input('Type a sequence of numbers or X to exit: ')
        if sequence.upper() == 'X':
            return None

        array = np.array([int(x) for x in sequence if x.isdigit()])
        sequence_size = len(array)
        if sequence_size > num_time_steps:
            print('Use at most %d digits', num_time_steps)
            continue

        ready = True

    return array


def _prepare_to_feed(sequence, num_time_steps):
    """
    Prepare a simple array to be fed to the autoencoders.

    It adds necessary padding and changes the dimensions as needed.
    """
    padding_dims = (0, num_time_steps - len(sequence))
    padded = np.pad(sequence, padding_dims, 'constant', constant_values=Symbol.END)

    # we have to have the shape (time_steps, batch_size)
    return np.array([padded]).T


def interactive_memorizer(model, max_time_steps):
    """
    Read an input from the prompt and run the memorizer.
    :param model: the autoencoder
    :param max_time_steps: maximum number of time steps a sequence
        can have
    :return: the memorizer answer or None
    """
    sequence = _read_sequence_from_prompt(max_time_steps)
    if sequence is None:
        return None

    sequence_len = len(sequence)
    sequence = _prepare_to_feed(sequence, max_time_steps)

    answer = model.run(sess, sequence, [sequence_len])[:, 0]
    return answer


def interactive_sum(model, max_time_steps):
    """
    Read an input from the prompt and run the sum autoencoder.
    :param model: the autoencoder
    :param max_time_steps: maximum number of time steps a sequence
        can have
    :return: the sum answer or None
    """
    sequence1 = _read_sequence_from_prompt(max_time_steps)
    if sequence1 is None:
        return None
    sequence2 = _read_sequence_from_prompt(max_time_steps)
    if sequence2 is None:
        return None

    sequence1_len = len(sequence1)
    sequence2_len = len(sequence2)
    sequence1 = _prepare_to_feed(sequence1, max_time_steps)
    sequence2 = _prepare_to_feed(sequence2, max_time_steps)

    answer = model.run(sess, sequence1, sequence2,
                       [sequence1_len], [sequence2_len])[:, 0]
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('type', help='Which autoencoder to evaluate',
                        choices=['memorizer', 'sum'])
    parser.add_argument('load', help='File to load the model from')
    args = parser.parse_args()

    params = utils.load_parameters(args.load)
    num_time_steps = params['num_time_steps']
    embedding_size = params['embedding_size']

    sess = tf.Session()

    if args.type == 'sum':
        autoencoder_class = addition.AdditionAutoEncoder
        interactive_function = interactive_sum
    else:
        autoencoder_class = memorizer.MemorizerAutoEncoder
        interactive_function = interactive_memorizer

    model = autoencoder_class(embedding_size, num_time_steps, None)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, args.load)

    while True:
        answer = interactive_function(model, num_time_steps)
        if answer is None:
            break
        answer_str = ' '.join(str(digit) for digit in answer)

        print('Model output: ', answer_str)

