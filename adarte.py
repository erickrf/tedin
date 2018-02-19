# -*- coding: utf-8 -*-

"""
Python implementation of the Adarte algorithm for recognizing textual
entailment.
"""

from __future__ import division, print_function, unicode_literals

import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from infernal import utils
from infernal import adarte_utils as adarte


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs (TSV)')
    parser.add_argument('model', help='Directory to save model')
    args = parser.parse_args()

    pairs = utils.load_pickled_pairs(args.train)
    label_dict = utils.create_label_dict(pairs)

    bags = [adarte.get_bag_of_transformations(pair) for pair in pairs]
    transformation_dict = adarte.create_transformation_dict(bags)
    x = adarte.convert_bags(bags, transformation_dict)
    y = adarte.get_labels(pairs, label_dict)

    # c = RandomForestClassifier(100)
    c = SVC()
    c.fit(x, y)
    acc = c.score(x, y)
    print('Training set accuracy: {}'.format(acc))

    adarte.save_model(args.model, c, label_dict, transformation_dict)
