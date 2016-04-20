# -*- coding: utf-8 -*-

"""
Utilities for working with the numeric LSTM.
"""

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
    exps = np.exp(possible_sizes)
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

