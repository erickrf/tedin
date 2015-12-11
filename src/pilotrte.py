# -*- coding: utf-8 -*-

'''
Simple script for RTE. It extracts a few features from the input
T and H, following Sharma et al. 2015 (NAACL) and apply supervised
classification.

This script will train a new model.
'''

import argparse
import os
import cPickle

import utils
import feature_extraction as fe


def train_models(pairs):
    '''
    Train a classifier with the given pairs
    '''
    x, y, z = fe.pipeline_minimal(pairs)
    classifier = utils.train_classifier(x, y)
    regressor = utils.train_regressor(x, z)
    
    return classifier, regressor

def save_model(model, dirname, filename):
    '''
    Save a model with pickle in the given path.
    '''
    output_file = os.path.join(dirname, filename)
    with open(output_file, 'wb') as f:
        cPickle.dump(model, f, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='RTE XML file for training')
    parser.add_argument('output_dir', help='Directory to save models')
    args = parser.parse_args()
    
    pairs = utils.read_xml(args.input)
    classifier, regressor = train_models(pairs)
    
    save_model(classifier, args.output_dir, 'classifier.dat')
    save_model(regressor, args.output_dir, 'regressor.dat')

    
    