# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

"""
Utility functions for training and running adarte models
"""

from six.moves import cPickle
from six.moves import reduce
import numpy as np
import zss
import os

from . import feature_extraction as fe
from . import utils


MODEL = 'model.pickle'
LABEL_DICT = 'label-dict.json'
TRANSFORMATION_DICT = 'transformation-dict.json'


def create_transformation_dict(transformations):
    """
    Create a transformation dictionary mapping all the combinations of
    transformation type and respective arguments to integers.

    :param transformations: list of sets with the bag of transformations for
        each pair in a dataset
    """
    all_transformations = reduce(lambda s1, s2: s1.union(s2), transformations)
    transformation_dict = {trans: i
                           for i, trans in enumerate(all_transformations)}

    return transformation_dict


def convert_bags(transformations, transformation_dict):
    """
    Convert a list of bag of transformations into a numpy 2d array representing
    the pairs with one-hot encoding.

    :param transformations: list of sets
    :param transformation_dict: dictionary mapping transformations to integers
    :return: 2d numpy array
    """
    num_pairs = len(transformations)
    num_operations = len(transformation_dict)
    matrix = np.zeros([num_pairs, num_operations], np.int32)

    for i, bag in enumerate(transformations):
        for transformation in bag:
            if transformation not in transformation_dict:
                # transformation unknown at training time, we don't know
                # what to make out of it
                continue

            j = transformation_dict[transformation]
            matrix[i, j] = 1

    return matrix


def get_bag_of_transformations(pair):
    """
    Return the set of operations representing the pair, except for matches.

    The returned set has tuples (operation, arg1 dep_rel, arg2 dep_rel)
    In the case of insert and remove, one of the latter is None.
    """
    cost, ops = fe.simple_tree_distance(pair, return_operations=True)
    bag = set()
    for op in ops:
        op_type = op.type
        if op_type == zss.Operation.match:
            continue

        dep_rel1 = op.arg1.dep_index if op.arg1 else None
        dep_rel2 = op.arg2.dep_index if op.arg2 else None
        bag.add((op_type, dep_rel1, dep_rel2))

    return bag


def get_labels(pairs, label_dict):
    """
    Return a numpy 1d array with the labels of the pairs
    """
    return np.array([label_dict[pair.label.name] for pair in pairs])


def save_model(directory, model, label_dict, transformation_dict):
    """
    Save the given model and related data to the given directory.
    """
    model_path = os.path.join(directory, MODEL)
    label_dict_path = os.path.join(directory, LABEL_DICT)
    trans_dict_path = os.path.join(directory, TRANSFORMATION_DICT)

    with open(model_path, 'wb') as f:
        cPickle.dump(model, f, -1)

    utils.write_label_dict(label_dict, label_dict_path)

    with open(trans_dict_path, 'wb') as f:
        cPickle.dump(transformation_dict, f, -1)


def _load_pickle(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj


def load_model(directory):
    model_path = os.path.join(directory, MODEL)
    return _load_pickle(model_path)


def load_label_dict(directory):
    label_dict_path = os.path.join(directory, LABEL_DICT)
    return utils.load_label_dict(label_dict_path)


def load_transformation_dict(directory):
    trans_dict_path = os.path.join(directory, TRANSFORMATION_DICT)
    return _load_pickle(trans_dict_path)


def convert_pairs(pairs, transformation_dict):
    """
    Convert the pairs to a numpy 2d array representation
    """
    bags = [get_bag_of_transformations(pair) for pair in pairs]
    return convert_bags(bags, transformation_dict)
