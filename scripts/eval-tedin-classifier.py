# -*- coding: utf-8 -*-

"""
Evaluate a TEDIN classifier.
"""

from __future__ import division, print_function, unicode_literals

import argparse
from sklearn.metrics import f1_score

from infernal import utils
from infernal import nn


def eval_performance(gold_labels, sys_labels):
    macro_f1 = f1_score(gold_labels, sys_labels, average='macro')
    accuracy = (gold_labels == sys_labels).sum() / len(gold_labels)
    print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with trained model')
    parser.add_argument('dep_dict',
                        help='Dictionary of dependency labels (JSON)')
    parser.add_argument('embeddings', help='Numpy file with embeddings')
    parser.add_argument('data', help='Preprocessed pickled test pairs')
    parser.add_argument('--lower', help='Lowercase tokens', action='store_true')
    args = parser.parse_args()

    extra_embeddings_path = utils.get_embeddings_path(args.model)
    wd_path = utils.get_vocabulary_path(args.embeddings)
    wd = utils.load_vocabulary(wd_path)
    embeddings = utils.load_embeddings([args.embeddings, extra_embeddings_path])

    label_dict = utils.load_label_dict(args.model)
    dep_dict = utils.load_label_dict(args.dep_dict)
    data, label_dict = utils.load_tedin_data(args.data, wd, dep_dict,
                                             label_dict, args.lower)
    tedin = nn.TreeEditDistanceNetwork.load(args.model, embeddings)
    answers = tedin.classify(data)
    gold_labels = data.labels

    eval_performance(gold_labels, answers)
