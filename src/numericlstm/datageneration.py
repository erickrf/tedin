# -*- coding: utf-8 -*-

from __future__ import division

"""
Functions for automatic generating data for the numeric autoencoders.
"""

import numpy as np

import utils
from utils import Symbol


def int_to_array(x):
    """
    This is slower than array to int
    """
    if x == 0:
        return np.array([0])

    digits = []
    while x > 0:
        digits.insert(0, x % 10)
        x /= 10

    return np.array(digits)


def batch_array_to_ints(x):
    """
    Return an array containing the integers represented by each
    row of the input array.
    """
    total = np.zeros_like(x[0])
    for row in x:
        total = 10 * total + row
    return total


def array_to_int(x):
    total = 0
    for i in x:
        total = 10 * total + i
    return total


def ints_to_array_batch(x, max_time_steps):
    """
    Takes an integer 1-D array and returns a 2-D array with their digits.

    The most significant figures are at the index 0

    :param x: 1-D integer array
    :param max_time_steps: the maximum possible number of time steps
    :return: 2-D array with shape (max_time_steps, len(x)) and one digit per cell.
    """
    x = np.copy(x)
    max_power_10 = max_time_steps - 1

    # this is the maximum number allowed
    max_possible_number = 10 ** max_time_steps - 1
    assert not any(x > max_possible_number), 'Cannot represent a number in the array'

    new = np.full((max_time_steps, len(x)), Symbol.END, np.int32)
    # just to make sure in case of the results is 0
    new[0] = 0

    # start from 2 so that the last row always has END
    for i in range(1, max_time_steps + 1):
        big_ones = x >= (10 ** max_power_10)
        new[-i][big_ones] = x[big_ones] % 10
        x[big_ones] //= 10

        # on a related note: raising a variable to a power is faster than
        # dividing it by 10
        max_power_10 -= 1

    return new


def ints_to_array_batch_reversed(x):
    x = np.copy(x)
    new = np.full((8, len(x)), 10, np.int32)
    # just to make sure in case the result is 0
    new[0] = 0

    for i in range(8):
        new[i][x > 0] = x[x > 0] % 10
        x /= 10

    return new


def generate_numbers(num_time_steps, num_padding, batch_size):
    """
    Generate a 2-D array containing digits, and a corresponding
    array with their concatenation.

    Each number will have the determined number of valid time steps,
    plus the given number of padding (as the END symbol)
    """
    shape = (num_time_steps, batch_size)
    digits = np.random.randint(0, 10, shape)
    numbers = batch_array_to_ints(digits)
    padding = np.full((num_padding, batch_size),
                      Symbol.END, np.int32)
    digits = np.concatenate([digits, padding])

    return (digits, numbers)


def generate_addition_data(num_time_steps, num_items):
    """
    Generate training data for the addition autoencoder

    :param num_time_steps: the maximum number of time steps of the encoder
        (it is understood that the result has one extra step)
    :param num_items: the number of items in the dataset (an item is
        the both terms and their result)
    :return: a tuple (first_terms, second_terms, first_term_sizes,
        second_term_sizes, results)
    """
    digits_pool = []
    number_pool = []

    # first, generate the terms. we generate twice as much as `num_items`
    # since each item has two of them
    sizes = np.arange(1, num_time_steps + 1)
    exps = 2 ** sizes
    proportions = 2 * num_items * exps / np.sum(exps)
    proportions = np.ceil(proportions).astype(np.int)

    # just to make sure the resulting dataset has the intended size
    # we discount from the last size, which has the most items
    total_after_ceil = np.sum(proportions)
    proportions[-1] -= (total_after_ceil - 2 * num_items)

    for i in range(num_time_steps):
        size = sizes[i]
        num_padding = num_time_steps - size
        num_sequences = np.int(np.ceil(proportions[i]))

        digits, numbers = generate_numbers(size, num_padding, num_sequences)
        digits_pool.append(digits)
        number_pool.append(numbers)

    all_digits = np.concatenate(digits_pool, 1)
    all_numbers = np.concatenate(number_pool)

    # shuffle both lists with the same permutation
    rng_state = np.random.get_state()
    # shuffle works row-wise, but we want to shuffle columns
    np.random.shuffle(all_digits.T)
    np.random.set_state(rng_state)
    np.random.shuffle(all_numbers)

    # now divide it in the middle to get the two terms
    middle = np.int(all_digits.shape[1] / 2)
    first_terms = all_digits[:, :middle]
    first_terms_values = all_numbers[:middle]
    second_terms = all_digits[:, middle:]
    second_terms_values = all_numbers[middle:]

    results_integer = first_terms_values + second_terms_values
    results = ints_to_array_batch(results_integer, num_time_steps+1)
    first_sizes = np.sum(first_terms != 10, 0)
    second_sizes = np.sum(second_terms != 10, 0)

    return (first_terms, second_terms,
            first_sizes, second_sizes,
            results)


def _generate_memorizer_dataset(array_size, num_sequences, return_sizes=True):
    """
    Generate one dataset as a 2-dim numpy array

    :param array_size: the array size expected by the network
    :param num_sequences: the total number of sequences (columns) in the result
    :param return_sizes: if True, returns a tuple with the dataset and
        a 1-d array with the size of each sequence
    """
    data = np.random.randint(0, 10, (array_size, num_sequences))
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


def generate_memorizer_data(train_size, valid_size, num_time_steps):
    """
    Generate data for training and validation, shuffle and return them.

    :return: a 4-tuple (train_set, train_sizes, valid_set, valid_sizes)
    """
    total_data = train_size + valid_size
    data, sizes = _generate_memorizer_dataset(num_time_steps, total_data)
    utils.shuffle_data_memorizer(data, sizes)

    # removing duplicates must be change to account for the sizes.... at any rate,
    # we were getting 5 duplicates out of 32k. i don't think we really need it
    # data = remove_duplicates(data)
    train_set = data[:, :train_size]
    valid_set = data[:, train_size:]
    train_sizes = sizes[:train_size]
    valid_sizes = sizes[train_size:]

    return (train_set, train_sizes, valid_set, valid_sizes)
