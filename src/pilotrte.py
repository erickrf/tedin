# -*- coding: utf-8 -*-

'''
Simple script for RTE. It extracts a few features from the input
and trains a classifier and regressor models.
'''

import argparse
import importlib
import logging
import os
import cPickle

import utils

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

def save_model(model, dirname, filename):
    '''
    Save a model with pickle in the given path.
    '''
    output_file = os.path.join(dirname, filename)
    with open(output_file, 'wb') as f:
        cPickle.dump(model, f, -1)

def set_log(verbose):
    '''
    Set the system-wide logging configuration.
    
    :type verbose: bool
    '''
    log_level = logging.INFO if verbose else logging.WARN
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=log_level)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='RTE XML file for training')
    parser.add_argument('output_dir', help='Directory to save models')
    parser.add_argument('configuration', 
                        help='Python file with system configuration (must be'\
                        'inside "configurations" directory)')
    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='Verbose mode')
    args = parser.parse_args()
    
    set_log(args.verbose)
    
    logging.info('Reading pairs from {}'.format(args.input))
    pairs = utils.read_xml(args.input)
    
    classifier, regressor = train_models(pairs, args.configuration)
    
    logging.info('Saving models')
    save_model(classifier, args.output_dir, 'classifier.dat')
    save_model(regressor, args.output_dir, 'regressor.dat')

    