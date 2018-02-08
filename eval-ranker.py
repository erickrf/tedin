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
    parser.add_argument('embeddings', help='Numpy file with embeddings')
    parser.add_argument('data', help='Preprocessed pickled test pairs')
    args = parser.parse_args()

    extra_embeddings_path = os.path.join(args.model, 'extra-embeddings.npy')
    embeddings = utils.load_embeddings([args.embeddings, extra_embeddings_path])
    data = utils.load_positive_and_negative_data(args.data)
    ranker = nn.PairRanker.load(args.model, embeddings)
    loss = ranker.evaluate(data)
    print('Computed loss: %f' % loss)
