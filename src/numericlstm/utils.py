# -*- coding: utf-8 -*-

from __future__ import division

"""
Utilities for working with the numeric autoencoders.
"""

import json
import os
import numpy as np


class Symbol(object):
    """
    Placeholder class for values used in the RNNs.
    """
    END = 10
    GO = 11


def generate_sequences(array_size, sequence_size, batch_size):
    """
    Generate a sequence of numbers as an array.

    All sequences must have the same size. The array is filled with
    END symbols as necessary.

    :param array_size: the array size expected by the encoder/decoder
    :param sequence_size: the number of items (digits) actually
        contained in the sequences. After that many entries, the array
        is filled with the END symbol.
    :param batch_size: number of sequences
    """
    dims = (array_size, batch_size)
    sequences = np.random.random_integers(0, 9, dims)
    sequences[sequence_size:] = Symbol.END

    return sequences


def remove_duplicates(x):
    """
    Return a copy of the array x without duplicate columns
    """
    order = np.lexsort(x)
    ordered = x[:, order]
    diffs = np.diff(ordered, axis=1)
    diff_sums = np.sum(np.abs(diffs), 0)
    unique_indices = np.ones(x.shape[1], dtype='bool')
    unique_indices[1:] = diff_sums != 0
    unique_x = ordered[:, unique_indices]

    return unique_x


def shuffle_data_and_sizes(data, sizes):
    """
    Convenient function for shuffling a dataset and its sizes with the same
    RNG state.
    """
    rng_state = np.random.get_state()
    np.random.shuffle(data.T)
    np.random.set_state(rng_state)
    np.random.shuffle(sizes)


def generate_dataset(array_size, num_sequences, return_sizes=True):
    """
    Generate one dataset as a 2-dim numpy array

    :param array_size: the array size expected by the network
    :param num_sequences: the total number of sequences (columns) in the result
    :param return_sizes: if True, returns a tuple with the dataset and
        a 1-d array with the size of each sequence
    """
    data = np.random.random_integers(0, 9, (array_size, num_sequences))
    seq_sizes = np.empty(num_sequences, dtype=np.int)

    possible_sizes = np.arange(1, array_size + 1)
    exps = 2 ** possible_sizes #np.exp(possible_sizes)
    proportions = exps / np.sum(exps) * num_sequences
    proportions = np.ceil(proportions).astype(np.int)

    last_idx = 0
    for i, prop in enumerate(proportions, 1):
        until_idx = last_idx + prop

        data[i:, last_idx:until_idx] = Symbol.END
        seq_sizes[last_idx:until_idx] = i

        last_idx = until_idx

    if return_sizes:
        return (data, seq_sizes)

    return data


def get_data(train_size, valid_size, num_time_steps):
    """
    Generate data for training and validation, shuffle and return them.

    :return: a 4-tuple (train_set, train_sizes, valid_set, valid_sizes)
    """
    total_data = train_size + valid_size
    data, sizes = generate_dataset(num_time_steps, total_data)
    shuffle_data_and_sizes(data, sizes)

    # removing duplicates must be change to account for the sizes.... at any rate,
    # we were getting 5 duplicates out of 32k. i don't think we really need it
    # data = remove_duplicates(data)
    train_set = data[:, :train_size]
    valid_set = data[:, train_size:]
    train_sizes = sizes[:train_size]
    valid_sizes = sizes[train_size:]

    return (train_set, train_sizes, valid_set, valid_sizes)


def save_parameters(basefilename, embedding_size, num_time_steps):
    """
    Save the arguments used to instantiate a model.

    :param basefilename: the base path to a filename. The suffix '-params.json'
        will be appended
    :param embedding_size: size of the embeddings
    :param num_time_steps: maximum number of time steps
    """
    filename = basefilename + '-params.json'
    data = {'embedding_size': embedding_size,
            'num_time_steps': num_time_steps}

    with open(filename, 'wb') as f:
        json.dump(data, f)


def load_parameters(basefilename):
    """
    Load a dictionary containing the parameters used to train an instance
    of the autoencoder.

    :param basefilename: the base path to the filename, without '-params.json'.
        It is the same base file used to save the tensorflow model.
    :return: a Python dictionary
    """
    filename = basefilename + '-params.json'
    with open(filename, 'rb') as f:
        data = json.load(f)

    return data


def get_accuracy(model, session, data, sizes, ignore_end=True):
    """
    Get the prediciton accuracy on the supplied data.

    :param model: an instance of the numeric LSTM
    :param session: current tensorflow session
    :param data: numpy array with shape (num_time_steps, batch_size)
    :param sizes: actual size of each sequence in data
    :param ignore_end: if True, ignore the END symbol
    """
    answer = model.run(session, data, sizes)

    # if the answer is longer than it should, truncate it
    if len(answer) > len(data):
        answer = answer[:len(data)]

    hits = answer == data
    total_items = answer.size

    if ignore_end:
        non_end = data != Symbol.END
        hits_non_end = hits[non_end]
        total_items = np.sum(non_end)

    acc = np.sum(hits_non_end) / total_items
    return acc
