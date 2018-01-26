# -*- coding: utf-8 -*-

"""
"""

from __future__ import division, print_function, unicode_literals

import argparse
import logging

from . import nn
from . import utils


def assign_word_indices(pairs, wd, lower=True):
    """
    Assign each token in the sentences of pairs their embedding index.

    Changes are in-place.

    This is done here instead of in a pre-processing stage to allow for
    different embedding models with different vocabularies.

    :param pairs: list of Pair objects
    :param wd: dictionary mapping strings to ints
    :param lower: whether to lowercase tokens before indexing
    """
    def get_index(token):
        if lower:
            return wd[token.lower()]
        return wd[token]

    for pair in pairs:
        for token in pair.annotated_t.tokens:
            token.index = get_index(token.text)

        for token in pair.annotated_h.tokens:
            token.index = get_index(token.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs')
    parser.add_argument('valid', help='Validation pairs')
    parser.add_argument('embeddings', help='Numpy embeddings file (txt file '
                                           ' with vocabulary will be sought)')
    parser.add_argument('model', help='Directory to save model and logs')
    parser.add_argument('-l', help='Learning rate', type=float,
                        dest='learning_rate', default=0.01)
    parser.add_argument('-n', help='Hidden units', type=int, default=100,
                        dest='num_units')
    parser.add_argument('-d', help='Dropout keep', type=float, dest='dropout',
                        default=1)
    parser.add_argument('-e', help='Number of steps', type=int, default=100,
                        dest='steps')
    parser.add_argument('-b', help='Batch size', type=int, default=16,
                        dest='batch')
    parser.add_argument('-f', help='Evaluation frequency', type=int, default=50,
                        dest='eval_frequency')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    wd, embeddings = utils.load_embeddings(args.embeddings)

    train_pairs = utils.read_pairs(args.train)
    valid_pairs = utils.read_pairs(args.valid)
    assign_word_indices(train_pairs, wd)
    assign_word_indices(valid_pairs, wd)

    train_data = nn.create_tedin_dataset(train_pairs, wd)
    valid_data = nn.create_tedin_dataset(valid_pairs, wd)

    params = nn.TedinParameters(args.learning_rate, args.dropout, args.batch,
                                args.steps)
    tedin = nn.TreeEditDistanceNetwork(args.num_units, embeddings.shape, 3)
    tedin.initialize(embeddings)
    tedin.train(train_data, valid_data, params, args.model, args.eval_frequency)
