# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Functions for reading external resources and 
providing easy access to them.
'''

import config
import logging

unitex_dictionary = None

# this dictionary maps tags from the Unitex tagset to the Universal
# dependencies one. Only these POS's need to be mapped
tag_map = {'V': 'VERB',
           'N': 'NOUN', 
           'A': 'ADJ'}

lemmatizable_tags = set(tag_map.values())

def _read_unitex_dictionary():
    '''
    Read the Unitex dictionary with inflected word forms. 
    
    This function ignores all entries with clitics because it
    is targeted at a system that uses Universal Dependencies and
    treats verbs and clitics separately, and removing clitics 
    saves memory. 
    '''
    global unitex_dictionary
    unitex_dictionary = {}
    
    logging.info('Reading Unitex dictionary')
    with open(config.unitex_dictionary_path, 'rb') as f:
        for line in f:
            line = unicode(line, 'utf-8').strip()
            # each line is in the format
            # inflected_word,lemma.POS:additional_morphological_metadata
            # the morphological metadata is only available for open class words
            inflected, rest = line.split(',')
            if '-' in inflected:
                continue
            
            lemma, morph = rest.split('.')
            if ':' in morph:
                pos, _ = morph.split(':')
            else:
                pos = morph
            
            if pos not in tag_map:
                continue
            
            ud_pos = tag_map[pos]
            unitex_dictionary[(inflected, ud_pos)] = lemma
            
    logging.info('Finished')

def get_lemma(word, pos):
    '''
    Retrieve the lemma of a word given its POS tag.
    
    If the combination of word and POS is not known, return the
    word itself. 
    '''
    global unitex_dictionary
    if unitex_dictionary is None:
        _read_unitex_dictionary()
    
    if pos not in lemmatizable_tags:
        return word
    
    word = word.lower()
    if (word, pos) in unitex_dictionary:
        return unitex_dictionary[(word, pos)]
    else:
        logging.warning('Could not find lemma for word {} with POS {}'.format(word, pos))
        return word
    