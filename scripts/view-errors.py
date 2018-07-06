# -*- coding: utf-8 -*-

"""
View the misclassified pairs by a shallow classifier.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np

from infernal import shallow_utils as shallow
from infernal import utils
from infernal import feature_extraction as fe


def get_misclassified_indices(classifier, normalizer, x, y):
    if normalizer:
        x = normalizer.transform(x)

    preds = classifier.predict(x)

    # np.where returns a tuple
    inds = np.where(preds != y)[0]
    wrong_preds = preds[inds]

    return inds, wrong_preds


def find_culprit_features(classifier, normalizer, x, y):
    """
    Run the classifier on the given data and report an analysis of the features
    responsible for the mistakes.

    :return: a tuple (inds, wrong_preds, scores)
        inds is an array with the indices of the wrongly classified pairs
        wrong_preds is an array with the wrong predictions
        scores is a 2d array with the score computed by the input feature *
            corresponding weight. The last entry in each row is the bias.
    """
    if normalizer:
        x = normalizer.transform(x)

    preds = classifier.predict(x)

    # np.where returns a tuple
    inds = np.where(preds != y)[0]

    # take the wrongly classified inputs and their predicted class
    x = x[inds]
    wrong_preds = preds[inds]

    # get the products of all input features by the weights, before summing
    w = classifier.coef_
    b = classifier.intercept_
    num_items, num_features = x.shape
    xw = x.reshape(num_items, 1, num_features) * w

    # include the bias as a last unit
    tiled = np.tile(b, [num_items, 1]).reshape([-1, b.shape[0], 1])
    xw_b = np.concatenate([xw, tiled], axis=2)

    # take the units corresponding to the chosen class
    xw_b = xw_b[np.arange(num_items), wrong_preds]

    # ranks is (num_wrong_tems, num_features)
    # reverse it so features are sorted from most to least relevant
    ranks = xw_b.argsort(1, )[:, ::-1]

    return inds, wrong_preds, xw_b


def load_exceptions(filename):
    premises = set()
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('T: '):
                premises.add(line.replace('T: ', '').strip())

    return premises


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
    fex = fe.FeatureExtractor(both=True)

    feature_names = fex.get_feature_names() + ['Bias']
    # inds, wrong_preds, relevances = find_culprit_features(
    #     classifier, normalizer, x, y)

    inds, preds = get_misclassified_indices(classifier, normalizer, x, y)
    ild = {ind: label for label, ind in ld.items()}
    for ind, pred in zip(inds, preds):
        pair = pairs[ind]
        gold_label = pair.entailment.name
        sys_label = ild[pred]

        # feature_inds = feature_relevances.argsort()[::-1][:5]
        # relevant_features = [feature_names[i] for i in feature_inds
        #                      if feature_relevances[i] > 0]

        print('T:', pair.t)
        print('H:', pair.h)
        print('Gold label:', gold_label, '\t\tSystem answer:', sys_label)
        print()
