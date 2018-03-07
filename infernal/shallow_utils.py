# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

"""
Utility functions for shallow models
"""

import os
import numpy as np
from six.moves import cPickle


CLASSIFIER_FILENAME = 'classifier.pickle'
NORMALIZED_FILENAME = 'normalizer.pickle'


def load_data(filename):
    """
    Load the data from the given path

    :return: a 2d array x and a 1d array y
    """
    data = np.load(filename)
    x, y = data['x'], data['y']
    return x, y


def load_classifier(path):
    """
    Load a classifer serialized as a pickle from the given directory.
    """
    path = os.path.join(path, CLASSIFIER_FILENAME)
    with open(path, 'rb') as f:
        classifier = cPickle.load(f)

    return classifier


def load_normalizer(path):
    """
    Load a normalizer serialized as a pickle from the given directory.
    """
    path = os.path.join(path, NORMALIZED_FILENAME)
    if not os.path.isfile(path):
        # no normalizer was created for this model
        return None

    with open(path, 'rb') as f:
        normalizer = cPickle.load(f)

    return normalizer


def save(path, classifier, normalizer=None):
    """
    Save the classifier and the normalizer to the given directory.
    """
    classifier_filename = os.path.join(path, CLASSIFIER_FILENAME)
    with open(classifier_filename, 'wb') as f:
        cPickle.dump(classifier, f)

    if normalizer:
        normalizer_filename = os.path.join(path, NORMALIZED_FILENAME)
        with open(normalizer_filename, 'wb') as f:
            cPickle.dump(normalizer, f)
