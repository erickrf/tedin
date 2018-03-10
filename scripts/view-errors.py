# -*- coding: utf-8 -*-

"""
View the misclassified pairs by a shallow classifier.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np

from infernal import shallow_utils as shallow
from infernal import utils


def get_misclassified_indices(classifier, normalizer, x, y):
    if normalizer:
        x = normalizer.transform(x)

    preds = classifier.predict(x)

    # np.where returns a tuple
    inds = np.where(preds != y)[0]
    wrong_preds = preds[inds]

    return inds, wrong_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='Preprocessed data (npz) to evaluate the'
                                     'classifier on')
    parser.add_argument('pairs', help='Pairs (pickle format) with sentences')
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('label_dict', help='Label dictionary')
    args = parser.parse_args()

    pairs = utils.load_pickled_pairs(args.pairs)
    ld = utils.load_label_dict(args.label_dict)
    x, y = shallow.load_data(args.data)
    classifier = shallow.load_classifier(args.model)
    normalizer = shallow.load_normalizer(args.model)

    inds, preds = get_misclassified_indices(classifier, normalizer, x, y)
    ild = {ind: label for label, ind in ld.items()}
    for ind, prediction in zip(inds, preds):
        pair = pairs[ind]
        print('T:', pair.t)
        print('H:', pair.h)
        gold = pair.entailment.name
        answer = ild[prediction]
        print('Gold label:', gold, '\t\tSystem answer:', answer)
        print()
