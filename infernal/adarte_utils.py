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

from . import utils
from . import openwordnetpt as own


MODEL = 'model.pickle'
LABEL_DICT = 'label-dict.json'
DEP_DICT = 'dep-dict.json'
TRANSFORMATION_DICT = 'transformation-dict.pickle'


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
    the pairs with multi-hot encoding.

    :param transformations: list of sets
    :param transformation_dict: dictionary mapping transformations to integers
    :return: 2d numpy array (num_pairs, num_possible_transformations)
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


def zhang_shasha(pair):
    """
    Compute a simple tree edit distance (TED) value and operations.

    Nodes are considered to match if they have the same dependency label and
    lemma (or are synonyms).
    """
    def get_children(node):
        return node.dependents

    def update_cost(node1, node2):
        if node1.dependency_relation == node2.dependency_relation and \
                own.are_synonyms(node1.lemma, node2.lemma):
            return 0
        return 1

    tree_t = pair.annotated_t
    tree_h = pair.annotated_h
    root_t = tree_t.root
    root_h = tree_h.root

    distance, operations = zss.distance(
        root_t, root_h, get_children, insert_cost=lambda _: 1,
        remove_cost=lambda _:1, update_cost=update_cost, return_operations=True)

    return distance, operations


def get_bag_of_transformations(pair, dep_dict):
    """
    Return the set of operations representing the pair, except for matches.

    The returned set has tuples (operation, arg1 dep_rel, arg2 dep_rel)
    In the case of insert and remove, one of the latter is None.
    """
    cost, ops = zhang_shasha(pair)
    bag = set()
    for op in ops:
        op_type = op.type
        if op_type == zss.Operation.match:
            continue

        dep_rel1 = op.arg1.dependency_relation if op.arg1 else None
        dep_rel2 = op.arg2.dependency_relation if op.arg2 else None
        dep_rel_ind1 = dep_dict[dep_rel1] if dep_rel1 in dep_dict else None
        dep_rel_ind2 = dep_dict[dep_rel2] if dep_rel2 in dep_dict else None
        bag.add((op_type, dep_rel_ind1, dep_rel_ind2))

    return bag


def get_labels(pairs, label_dict):
    """
    Return a numpy 1d array with the labels of the pairs
    """
    return np.array([label_dict[pair.entailment.name] for pair in pairs])


def save_model(directory, model, label_dict, dep_dict, transformation_dict):
    """
    Save the given model and related data to the given directory.
    """
    model_path = os.path.join(directory, MODEL)
    label_dict_path = os.path.join(directory, LABEL_DICT)
    trans_dict_path = os.path.join(directory, TRANSFORMATION_DICT)
    dep_dict_path = os.path.join(directory, DEP_DICT)

    with open(model_path, 'wb') as f:
        cPickle.dump(model, f, -1)

    utils.write_label_dict(label_dict, label_dict_path)
    utils.write_label_dict(dep_dict, dep_dict_path)

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


def load_dep_dict(directory):
    dep_dict_path = os.path.join(directory, DEP_DICT)
    return utils.load_label_dict(dep_dict_path)


def load_transformation_dict(directory):
    trans_dict_path = os.path.join(directory, TRANSFORMATION_DICT)
    return _load_pickle(trans_dict_path)


def convert_pairs(pairs, dep_dict, transformation_dict):
    """
    Convert the pairs to a numpy 2d array representation
    """
    bags = [get_bag_of_transformations(pair, dep_dict) for pair in pairs]
    return convert_bags(bags, transformation_dict)
