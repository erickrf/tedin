# -*- coding: utf-8 -*-

'''
Script to evaluate system performance on the ASSIN shared task data.

Author: Erick Fonseca
'''

from __future__ import division, print_function

import argparse
from xml.etree import cElementTree as ET
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, t, h, id_, entailment, similarity):
        '''
        :param entailment: boolean
        :param attribs: extra attributes to be written to the XML
        '''
        self.t = t
        self.h = h
        self.id = id_
        self.entailment = entailment
        self.similarity = similarity
    
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
        id_ = int(attribs['id'])
        
        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()
            
            if ent_string == 'none':
                ent_value = 0
            elif ent_string == 'entailment':
                ent_value = 1
            elif ent_string == 'paraphrase':
                ent_value = 2
            else:
                msg = 'Unexpected value for attribute "entailment" at pair {}: {}'
                raise ValueError(msg.format(id_, ent_value))
                        
        else:
            ent_value = None
            
        if 'similarity' in attribs:
            similarity = float(attribs['similarity']) 
        else:
            similarity = None
        
        if similarity is None and ent_value is None:
            msg = 'Missing both entailment and similarity values for pair {}'.format(id_)
            raise ValueError(msg)
        
        pair = Pair(t, h, id_, ent_value, similarity)
        pairs.append(pair)
    
    return pairs

def eval_rte(pairs_gold, pairs_sys):
    '''
    Evaluate the RTE output of the system against a gold score. 
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].entailment is None:
        print()
        print('No RTE output to evaluate')
        return
    
    gold_values = np.array([p.entailment for p in pairs_gold])
    sys_values = np.array([p.entailment for p in pairs_sys])
    macro_f1 = f1_score(gold_values, sys_values, average='macro')
    accuracy = (gold_values == sys_values).sum() / len(gold_values)
    
    print()
    print('RTE evaluation')
    print('Accuracy\tMacro F1')
    print('--------\t--------')
    print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))

def eval_similarity(pairs_gold, pairs_sys):
    '''
    Evaluate the semantic similarity output of the system against a gold score. 
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].similarity is None:
        print()
        print('No similarity output to evaluate')
        return
    
    gold_values = np.array([p.similarity for p in pairs_gold])
    sys_values = np.array([p.similarity for p in pairs_sys])
    pearson = pearsonr(gold_values, sys_values)[0]
    absolute_diff = gold_values - sys_values
    mse = (absolute_diff ** 2).mean()
    
    print()
    print('Similarity evaluation')
    print('Pearson\t\tMean Squared Error')
    print('-------\t\t------------------')
    print('{:7.2f}\t\t{:18.2f}'.format(pearson, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold_file', help='Gold file')
    parser.add_argument('system_file', help='File produced by a system')
    args = parser.parse_args()
    
    pairs_gold = read_xml(args.gold_file)
    pairs_sys = read_xml(args.system_file)
    
    eval_rte(pairs_gold, pairs_sys)
    eval_similarity(pairs_gold, pairs_sys)
