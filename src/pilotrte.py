# -*- coding: utf-8 -*-

'''
Simple script for RTE. It extracts a few features from the input
T and H, following Sharma et al. 2015 (NAACL) and apply supervised
classification.

This script will train a new model.
'''

import argparse
import datastructures
import xml.etree.cElementTree as ET

def read_xml(filename):
    '''
    Read an RTE XML file and return a list of Pair objects.
    '''
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        entailment = attribs['entailment'].lower() in ['yes', 'entailment']
        pair = datastructures.Pair(t, h, entailment, attribs)
        pairs.append(pair)
    
    return pairs

def train(pairs):
    '''
    Train a classifier with the given pairs
    '''
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='RTE XML file for training')
    args = parser.parse_args()
    
    pairs = read_xml(args.input)
    train(pairs)
    