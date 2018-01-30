# -*- coding: utf-8 -*-

'''
Script to read a data file from the SICK dataset and convert 
it to the RTE XML format.
'''

import argparse
import xml.etree.cElementTree as ET

from datastructures import Pair

def write_rte(filename, pairs):
    '''
    Write the given pairs to a file in RTE XML format.
    '''
    root = ET.Element('entailment-corpus')
    
    for pair in pairs:
        xml_pair = ET.SubElement(root, 'pair',
                                 score=pair.score,
                                 id=pair.id)
        
        xml_pair.attrib['entailment'] = 'ENTAILMENT' if pair.entailment else 'NONENTAILMENT'
        
        xml_t = ET.SubElement(xml_pair, 't')
        xml_t.text = pair.t
        xml_h = ET.SubElement(xml_pair, 'h')
        xml_h.text = pair.h
    
    tree = ET.ElementTree(root)
    tree.write(filename, 'utf-8')

def read_sick(filename):
    '''
    Read a SICK file and return a list of Pairs
    '''
    with open(filename, 'rb') as f:
        text = unicode(f.read(), 'utf-8')
    
    lines = text.splitlines()
    pairs = []
    
    # discard first line (field names)
    for i, line in enumerate(lines[1:], 1):
        fields = line.split('\t')
        try:
            pair_id, t, h, score, entailment = fields
        except ValueError:
            msg = 'Error reading file {} in line {}. Skipping line.'
            print msg.format(filename, i)
            continue
        
        # use a boolean value (contradiction merged with non-entailment)
        #TODO: use some kind of enum to set contradiction cases apart from non-entailment
        entailment = entailment == 'ENTAILMENT'
        
        pair = Pair(t, h, entailment)
        pair.score = score
        pair.id = pair_id
        pairs.append(pair)
    
    return pairs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    
    pairs = read_sick(args.input)
    write_rte(args.output, pairs)
