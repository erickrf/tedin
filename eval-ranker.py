# -*- coding: utf-8 -*-

"""
Script to evaluate the performance of a Pair Ranker.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import os

from infernal import utils
from infernal import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with trained model')
    parser.add_argument('dep_dict', help='Dependency label dictionary (JSON)')
    parser.add_argument('embeddings', help='Numpy file with embeddings')
    parser.add_argument('data', help='Preprocessed pickled test pairs')
    parser.add_argument('--lower', action='store_true', help='Lowercase tokens')
    args = parser.parse_args()

    extra_embeddings_path = utils.get_embeddings_path(args.model)
    wd_path = utils.get_vocabulary_path(args.embeddings)
    wd = utils.load_vocabulary(wd_path)
    dd = utils.load_label_dict(args.dep_dict)
    embeddings = utils.load_embeddings([args.embeddings, extra_embeddings_path])

    data = utils.load_positive_and_negative_data(args.data, wd, dd,
                                                 lower=args.lower)
    ranker = nn.PairRanker.load(args.model, embeddings)
    loss = ranker.evaluate(data)
    print('Computed loss: %f' % loss)
