# -*- coding: utf-8 -*-

'''
Simple script for RTE. It extracts a few features from the input
T and H, following Sharma et al. 2015 (NAACL) and apply supervised
classification.

This script will train a new model.
'''

import argparse

import utils


def train(pairs):
    '''
    Train a classifier with the given pairs
    '''
    utils.preprocess(pairs)
    
#     actual_train(pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='RTE XML file for training')
    args = parser.parse_args()
    
    pairs = utils.read_xml(args.input)
    train(pairs)
    