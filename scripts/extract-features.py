# -*- coding: utf-8 -*-

"""
Extract features to be later used by shallow classifiers.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np

from infernal import utils
from infernal import feature_extraction as fe
from infernal import openwordnetpt as own
from infernal import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='Preprocessed data in pickle format')
    parser.add_argument('embeddings_path',
                        help='Path to embeddings in npy format (there must be '
                             'a .txt file with vocabulary in the same folder)')
    parser.add_argument('output', help='npz file to save data')
    args = parser.parse_args()

    pairs = utils.load_pickled_pairs(args.data)
    stopwords = utils.load_stopwords()
    own.load_wordnet(config.ownpt_path)

    vocab_path = utils.get_vocabulary_path(args.embeddings_path)
    ed = utils.EmbeddingDictionary(vocab_path, args.embeddings_path)
    fex = fe.FeatureExtractor(True, stopwords, ed)

    feature_names = fex.get_feature_names()
    x = fex.extract_dataset_features(pairs)
    y = utils.extract_classes(pairs)

    np.savez(args.output, x=x, y=y)
