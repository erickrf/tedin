# -*- coding: utf-8 -*-

'''
Simple script for RTE. It extracts a few features from the input
and trains a classifier and regressor models.
'''

from __future__ import absolute_import

import argparse
import importlib
import logging
from collections import Counter
import sklearn
from six.moves import cPickle

import utils
import pipelines


def train_models(pairs, config_file):
    '''
    Train a classifier with the given pairs
    '''
    module_full_name = 'configurations.{}'.format(config_file)
    model_config = importlib.import_module(module_full_name)
    
    logging.info('Extracting features')
    x = model_config.extract_features(pairs, model_config)
    y = utils.extract_classes(pairs)
    z = utils.extract_similarities(pairs)
    
    logging.info('Training classifier')
    classifier = model_config.classifier_class(**model_config.classifier_parameters)
    classifier.fit(x, y)
    
    logging.info('Training regressor')
    regressor = model_config.regressor_class(**model_config.regressor_parameters)
    regressor.fit(x, z)
     
    return classifier, regressor


def set_log(verbose):
    '''
    Set the system-wide logging configuration.
    
    :type verbose: bool
    '''
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=log_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Preprocessed training data')
    parser.add_argument('output_dir', help='Directory to save models')
    parser.add_argument('pipeline', help='Which pipeline to use',
                        choices=['dependency', 'overlap'])
    parser.add_argument('-s', help='Use stopwords', action='store_true',
                        dest='use_stopwords')
    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='Verbose mode')
    parser.add_argument('--class', help='Choose classifier', default='maxent',
                        choices=['maxent', 'svm'], dest='classifier')
    parser.add_argument('--add-inv', help='Augment training set with inverted pairs',
                        action='store_true', dest='add_inv')
    parser.add_argument('--bin', action='store_true',
                        help='Binarize problem (change paraphrase to entailment)')
    args = parser.parse_args()
    
    set_log(args.verbose)
    pairs = utils.read_pairs(args.input, args.add_inv, args.bin)

    class_count = Counter(p.entailment for p in pairs)
    logging.debug('Read {} pairs'.format(len(pairs)))
    logging.debug('Class distribution: {}'.format(class_count))

    if args.classifier == 'maxent':
        classifier_class = sklearn.linear_model.LogisticRegression
    elif args.classifier == 'svm':
        classifier_class = sklearn.svm.SVC

    stopwords = utils.load_stopwords() if args.use_stopwords else None
    pipeline_class = pipelines.get_pipeline(args.pipeline)
    pipeline = pipeline_class(stopwords=stopwords,
                              classifier_class=classifier_class,
                              classifier_parameters={'class_weight': 'balanced'})
    assert isinstance(pipeline, pipelines.BaseConfiguration)

    logging.info('Training models')
    pipeline.train_classifier(pairs)

    logging.info('Saving models')
    pipeline.save(args.output_dir)

