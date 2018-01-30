# -*- coding: utf-8 -*-

"""
Train an unsupervised pair ranker that learns weights for tree edit distance
operations.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import logging

from infernal import utils
from infernal import nn
from infernal import datastructures as ds


def split_paraphrase_neutral(pairs):
    """
    Split a list of pairs into two lists: one containing only paraphrases
    and the other containing only neutral pairs.

    :return: tuple (paraphases, neutrals)
    """
    paraphrases = [pair for pair in pairs
                   if pair.entailment == ds.Entailment.paraphrase]
    neutrals = [pair for pair in pairs
                if pair.entailment == ds.Entailment.none]

    return paraphrases, neutrals


def load_pairs(path, wd, label_dict):
    """
    Load a pickle file with pairs and do some necessary preprocessing.

    :param path: path to saved pairs
    :param wd: word dictionary
    :param label_dict: label dictionary
    :return: tuple of ds.Datasets (positive, negative)
    """
    pairs = utils.read_pairs(path)
    pos_pairs, neg_pairs = split_paraphrase_neutral(pairs)
    pos_data = nn.create_tedin_dataset(pos_pairs, wd, label_dict)
    neg_data = nn.create_tedin_dataset(neg_pairs, wd, label_dict)

    msg = '%d positive and %d negative pairs' % (len(pos_data), len(neg_data))
    logging.info(msg)

    return pos_data, neg_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs')
    parser.add_argument('valid', help='Validation pairs')
    parser.add_argument('embeddings', help='Numpy embeddings file (txt file '
                                           ' with vocabulary will be sought)')
    parser.add_argument('label_dict', help='Dictionary with node labels')
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    wd, embeddings = utils.load_embeddings(args.embeddings)
    label_dict = utils.load_label_dict(args.label_dict)
    train_data = load_pairs(args.train, wd, label_dict)
    valid_data = load_pairs(args.valid, wd, label_dict)

    label_emb_shape = [len(label_dict), args.label_embedding_size]
    params = nn.TedinParameters(args.learning_rate, args.dropout, args.batch,
                                args.steps, args.num_units, embeddings.shape,
                                label_emb_shape, 3)

    ranker = nn.PairRanker(params)
    ranker.initialize(embeddings)
    ranker.train(train_data, valid_data, params,
                 args.model, args.eval_frequency)
