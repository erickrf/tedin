# -*- coding: utf-8 -*-

"""
"""

from __future__ import division, print_function, unicode_literals

import argparse
import logging
import os

from infernal import nn
from infernal import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='Training pairs (pickle)')
    parser.add_argument('valid', help='Validation pairs (pickle)')
    parser.add_argument('dep_dict',
                        help='Dictionary of dependency labels (JSON)')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('model', help='Directory to load pretrained model and '
                                      'save new version and logs')
    parser.add_argument('--lower', help='Lowercase tokens', action='store_true')
    parser.add_argument('-l', help='Learning rate', type=float,
                        dest='learning_rate', default=0.01)
    parser.add_argument('-d', help='Dropout keep', type=float, dest='dropout',
                        default=1)
    parser.add_argument('-e', help='Number of steps', type=int, default=100,
                        dest='steps')
    parser.add_argument('-b', help='Batch size', type=int, default=16,
                        dest='batch')
    parser.add_argument('-f', help='Evaluation frequency', type=int, default=50,
                        dest='eval_frequency')
    parser.add_argument('--label-weights', action='store_true',
                        dest='use_weights', help='Use label weights to counter '
                                                 'class imbalance')
    parser.add_argument('--load-label-dict', action='store_true',
                        help='Load label dictionary from the model directory',
                        dest='load_label_dict')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    utils.print_cli_args()

    wd_path = utils.get_vocabulary_path(args.embeddings)
    wd = utils.load_vocabulary(wd_path)
    extra_path = utils.get_embeddings_path(args.model)
    embeddings = utils.load_embeddings([args.embeddings, extra_path])
    dep_dict = utils.load_label_dict(args.dep_dict)

    if args.load_label_dict:
        label_dict = utils.load_label_dict(args.model)
    else:
        label_dict = None

    train_data, label_dict = utils.load_tedin_data(
        args.train, wd, dep_dict, label_dict, args.lower,
        use_weights=args.use_weights)
    valid_data, _ = utils.load_tedin_data(
        args.valid,  wd, dep_dict, label_dict, args.lower)

    utils.write_label_dict(label_dict,
                           os.path.join(args.model, 'label-dict.json'))

    tedin = nn.TreeEditDistanceNetwork.load(args.model, embeddings)
    params = tedin.params
    new_param_values = {'learning_rate': args.learning_rate,
                        'dropout': args.dropout,
                        'batch_size': args.batch,
                        'num_steps': args.steps}
    params.override_from_dict(new_param_values)
    nn.print_parameters()
    tedin.train(train_data, valid_data, args.model, args.eval_frequency)
