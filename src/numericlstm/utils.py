# -*- coding: utf-8 -*-

from __future__ import division

"""
Utilities for working with the numeric autoencoders.
"""

import json
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


def compute_accuracy(gold, answer, ignore_end=True):
    """
    Compute the model accuracy with the given data.
    :param gold: numpy array with shape (num_time_steps, batch_size)
    :param answer: system answer in the same format as `gold`
    :param ignore_end: if True, ignore the END symbol
    :return: the accuracy as a floating point number
    """
    # if the answer is longer than it should, truncate it
    if len(answer) > len(gold):
        answer = answer[:len(gold)]
    # or the opposite
    total_items = gold.size
    if len(gold) > len(answer):
        gold = gold[:len(answer)]

    hits = answer == gold
    if ignore_end:
        non_end = gold != Symbol.END
        hits_non_end = hits[non_end]
        total_items = np.sum(non_end)

    acc = np.sum(hits_non_end) / total_items
    return acc


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


def shuffle_data_addition(first_term, second_term, first_sizes, second_sizes, results):
    """
    Shuffle the data used in the addition task with the same RNG state.
    """
    rng_state = np.random.get_state()
    np.random.shuffle(first_term.T)
    np.random.set_state(rng_state)
    np.random.shuffle(second_term.T)
    np.random.set_state(rng_state)
    np.random.shuffle(first_sizes)
    np.random.set_state(rng_state)
    np.random.shuffle(second_sizes)
    np.random.set_state(rng_state)
    np.random.shuffle(results.T)


def shuffle_data_memorizer(data, sizes):
    """
    Convenient function for shuffling a dataset and its sizes with the same
    RNG state.
    """
    rng_state = np.random.get_state()
    np.random.shuffle(data.T)
    np.random.set_state(rng_state)
    np.random.shuffle(sizes)

