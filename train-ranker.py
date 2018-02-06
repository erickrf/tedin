# -*- coding: utf-8 -*-

"""
Train an unsupervised pair ranker that learns weights for tree edit distance
operations.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import logging
import numpy as np
import tensorflow as tf

from infernal import utils
from infernal import nn
from infernal import datastructures as ds


def split_positive_negative(pairs):
    """
    Split a list of pairs into two lists: one containing only positive
    and the other containing only negative pairs.

    :return: tuple (positives, negatives)
    """
    positive = [pair for pair in pairs
                if pair.entailment == ds.Entailment.entailment
                or pair.entailment == ds.Entailment.paraphrase]
    neutrals = [pair for pair in pairs
                if pair.entailment == ds.Entailment.none]

    return positive, neutrals


def load_pairs(path):
    """
    Load a pickle file with pairs and do some necessary preprocessing.

    :param path: path to saved pairs in pickle format
    :return: tuple of ds.Datasets (positive, negative)
    """
    pairs = utils.read_pickled_pairs(path)
    pos_pairs, neg_pairs = split_positive_negative(pairs)
    pos_data = nn.create_tedin_dataset(pos_pairs)
    neg_data = nn.create_tedin_dataset(neg_pairs)

    msg = '%d positive and %d negative pairs' % (len(pos_data), len(neg_data))
    logging.info(msg)

    return pos_data, neg_data


def print_variables():
    """
    Print the defined tensorflow variables
    """
    print('Tensorflow variables:')
    for v in tf.global_variables():
        print(v.name, v.shape.as_list())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs')
    parser.add_argument('valid', help='Validation pairs')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('model', help='Directory to save model and logs')
    parser.add_argument('-l', help='Learning rate', type=float,
                        dest='learning_rate', default=0.01)
    parser.add_argument('--le', help='Label embedding size', default=10,
                        type=int, dest='label_embedding_size')
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
    parser.add_argument('-r', help='Opoeration cost regularizer', type=float,
                        default=0, dest='cost_regularizer')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    utils.print_cli_args()

    embeddings = np.load(args.embeddings)
    label_dict = utils.load_label_dict(args.label_dict)
    train_data = load_pairs(args.train)
    valid_data = load_pairs(args.valid)

    label_emb_shape = [len(label_dict), args.label_embedding_size]
    params = nn.TedinParameters(args.learning_rate, args.dropout, args.batch,
                                args.steps, args.num_units, embeddings.shape,
                                label_emb_shape, 3, args.cost_regularizer)

    ranker = nn.PairRanker(params)
    ranker.initialize(embeddings)
    print_variables()
    ranker.train(train_data, valid_data, params,
                 args.model, args.eval_frequency)
