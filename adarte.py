# -*- coding: utf-8 -*-

"""
Python implementation of the Adarte algorithm for recognizing textual
entailment.
"""

from __future__ import division, print_function, unicode_literals

import argparse
from six.moves import cPickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

from infernal import utils
from infernal import openwordnetpt as own
from infernal import adarte_utils as adarte
from infernal import config


def load_or_create_label_dict(pairs, path):
    """
    If path is given, load a label dictionary. If not, create one from the data.
    """
    if path:
        return utils.load_label_dict(path)

    return utils.create_label_dict(pairs)


def load_or_create_dep_dict(pairs, path):
    """
    If path is given, load a label dictionary. If not, create one from the data.
    """
    if path:
        return utils.load_label_dict(path)

    return utils.create_dependency_dict(pairs)


def load_or_create_trans_dict(pairs, path):
    """
    If path is given, load a label dictionary. If not, create one from the data.
    """
    if path:
        with open(path, 'rb') as f:
            return cPickle.load(f)

    return adarte.create_transformation_dict(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs (pickle)')
    parser.add_argument('--model', help='Directory to save model. If not given,'
                                        ' no model is trained.')
    parser.add_argument('--save-data', dest='save_data',
                        help='If given, save the input and output data in numpy'
                             ' npz format.')
    parser.add_argument('--label-dict', help='If not given, one is created',
                        dest='label_dict')
    parser.add_argument('--dep-dict', help='If not given, one is created',
                        dest='dep_dict')
    parser.add_argument('--trans-dict', help='If not given, one is created',
                        dest='trans_dict')
    args = parser.parse_args()

    print('Loading data')
    pairs = utils.load_pickled_pairs(args.train)
    label_dict = load_or_create_label_dict(pairs, args.label_dict)
    dep_dict = load_or_create_dep_dict(pairs, args.dep_dict)

    print('Extracting transformations from pairs')
    own.load_wordnet(config.ownpt_path)
    bags = [adarte.get_bag_of_transformations(pair, dep_dict) for pair in pairs]
    trans_dict = load_or_create_trans_dict(pairs, args.trans_dict)

    print('Extracted %d different transformation features' % len(trans_dict))
    x = adarte.convert_bags(bags, trans_dict)
    y = adarte.get_labels(pairs, label_dict)
    if args.save_data:
        np.savez(args.save_data, x=x, y=y)

    if args.model is None:
        exit()

    c = LogisticRegression()
    c.fit(x, y)
    acc = c.score(x, y)
    print('Training set accuracy: {}'.format(acc))

    adarte.save_model(args.model, c, label_dict, dep_dict, trans_dict)
