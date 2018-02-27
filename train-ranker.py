# -*- coding: utf-8 -*-

"""
Train an unsupervised pair ranker that learns weights for tree edit distance
operations.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import logging
import tensorflow as tf

from infernal import utils
from infernal import nn


def print_variables():
    """
    Print the defined tensorflow variables
    """
    print('Tensorflow variables:')
    for v in tf.global_variables():
        print(v.name, v.shape.as_list())


def get_num_dep_labels(dataset):
    # shape of nodes is (num_items, num_nodes, 2)
    # last dim is [word_index, label_index]
    return dataset.nodes1[:, :, 1].max() + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs (preprocessed pickle)')
    parser.add_argument('valid', help='Validation pairs (same format)')
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

    # add a single OOV vector here
    # TODO: use a more sensible way of generating embeddings for OOV words
    embeddings = utils.load_embeddings(args.embeddings, add_vectors=1,
                                       dir_to_save=args.model)
    train_data = utils.load_positive_and_negative_data(args.train)
    valid_data = utils.load_positive_and_negative_data(args.valid)
    num_labels = get_num_dep_labels(train_data[0])

    label_emb_shape = [num_labels, args.label_embedding_size]
    params = nn.TedinParameters(args.learning_rate, args.dropout, args.batch,
                                args.steps, args.num_units, embeddings.shape,
                                label_emb_shape, 3, args.cost_regularizer)

    ranker = nn.PairRanker(params)
    ranker.initialize(embeddings)
    nn.print_parameters()
    ranker.train(train_data, valid_data, args.model, args.eval_frequency)
