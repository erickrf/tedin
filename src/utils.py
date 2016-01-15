# -*- coding: utf-8 -*-

'''
Utility functions
'''

import re
from xml.etree import cElementTree as ET
from nltk.tokenize import RegexpTokenizer
from xml.dom import minidom
import traceback
import logging
import nltk
import numpy as np

import config
import external
import datastructures


def tokenize_sentence(text, change_digits=False):
    '''
    Tokenize the given sentence in Portuguese. The tokenization is done in conformity
    with Universal Treebanks.
    
    :param change_digits: if True, replaces all digits with 9.
    '''
    if change_digits:
        text = re.sub(r'\d', '9', text)
    
    tokenizer_regexp = ur'''(?ux)
    # the order of the patterns is important!!
    (?:[Mm]\.?[Ss][Cc])\.?|           # M.Sc. with or without capitalization and dots
    (?:[Pp][Hh]\.?[Dd])\.?|           # Same for Ph.D.
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d{1,3}(?:\.\d{3})*(?:,\d+)|      # numbers in format 999.999.999,99999
    \d{1,3}(?:,\d{3})*(?:\.\d+)|      # numbers in format 999,999,999.99999
    \d+:\d+|                          # time and proportions
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    \$|                               # currency sign
    (?:[\#@]\w+])|                    # Hashtags and twitter user names
    \w+|                              # words with numbers
    -+|                               # any sequence of dashes
    \.{3,}|                           # ellipsis or sequences of dots
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)
    
    return tokenizer.tokenize(text)

    
def extract_classes(pairs):
    '''
    Extract the class infomartion (paraphrase, entailment, none, contradiction)
    from the pairs. 
    
    :return: a numpy array with values from 0 to num_classes - 1
    '''
    classes = np.array([pair.entailment.value - 1 for pair in pairs])
    return classes

def extract_similarities(pairs):
    '''
    Extract the similarity value from the pairs.
    
    :return: a numpy array
    '''
    z = np.array([p.similarity for p in pairs])
    return z

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
        
        # the entailment relation is expressed differently in some versions
        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()
            
            if ent_string in ['yes', 'entailment']:
                entailment = datastructures.Entailment.entailment
            elif ent_string == 'paraphrase':
                entailment = datastructures.Entailment.paraphrase
            elif ent_string == 'contradiction':
                entailment = datastructures.Entailment.contradiction
            else:
                entailment = datastructures.Entailment.none
                        
        elif 'value' in attribs:
            if attribs['value'].lower() == 'true':
                entailment = datastructures.Entailment.entailment
            else:
                entailment = datastructures.Entailment.none
            
        if 'similarity' in attribs:
            similarity = float(attribs['similarity']) 
        else:
            similarity = None
            
        pair = datastructures.Pair(t, h, entailment, similarity)
        pairs.append(pair)
    
    return pairs

def write_rte_file(filename, pairs, **attribs):
    '''
    Write an XML file containing the given RTE pairs.
    
    :param pairs: list of Pair objects
    :parma task: the task attribute in the XML elements
    '''
    root = ET.Element('entailment-corpus')
    for i, pair in enumerate(pairs, 1):
        xml_attribs = {'id':str(i)}
        
        # add any other attributes supplied in the function call or the pair
        xml_attribs.update(attribs)
        xml_attribs.update(pair.attribs)
        
        xml_pair = ET.SubElement(root, 'pair', xml_attribs)
        xml_t = ET.SubElement(xml_pair, 't', pair.t_attribs)
        xml_h = ET.SubElement(xml_pair, 'h', pair.h_attribs)
        xml_t.text = pair.t.strip()
        xml_h.text = pair.h.strip()
        
    # produz XML com formatação legível (pretty print)
    xml_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(xml_string)
    with open(filename, 'wb') as f:
        f.write(reparsed.toprettyxml('    ', '\n', 'utf-8'))

def preprocess_minimal(pairs):
    '''
    Apply a minimal preprocessing pipeline.
    It includes only a tokenizer.
    '''
    for pair in pairs:
        t = pair.t.lower()
        tokens = tokenize_sentence(t, False)
        s = datastructures.Sentence(None)
        s.tokens = tokens
        pair.annotated_t = s
        
        h = pair.h.lower()
        tokens = tokenize_sentence(h, False)
        s = datastructures.Sentence(None)
        s.tokens = tokens
        pair.annotated_h = s

def train_classifier(x, y):
    '''
    Train and return a classifier with the supplied data
    '''
    classifier = config.classifier_class(class_weight='auto')
    classifier.fit(x, y)
    
    return classifier

def train_regressor(x, y):
    '''
    Train and return a regression model (for similarity) with the supplied data.
    '''
    regressor = config.regressor_class()
    regressor.fit(x, y)
    
    return regressor

def preprocess_dependency(pairs):
    '''
    Preprocess the given pairs with a dependency parser.
    
    :param parser: which parser to use to preprocess. Allowed values
        are 'palavras', 'corenlp' and 'malt'
    '''
    if config.parser == 'corenlp':
        parser_function = external.call_corenlp
        parser_format = 'conll'
    elif config.parser == 'palavras':
        parser_function = external.call_palavras
        parser_format = 'palavras'
    elif config.parser == 'malt':
        parser_function = external.call_malt
        parser_format = 'conll'
    else:
        raise ValueError('Unknown parser: %s' % config.parser)
    
    for i, pair in enumerate(pairs):
        # run both at once to save time -- especially important if using a parser server
        texts = pair.t + '\n\n' + pair.h
        output = parser_function(texts)
        output_t, output_h = output.split('\n\n', 1)
        try:
            pair.annotated_t = datastructures.Sentence(output_t, parser_format)
            pair.annotated_h = datastructures.Sentence(output_h, parser_format)
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)
        
        pairs[i] = pair

def tokenize(text):
    """
    Return a list of lists of the tokens in the text. One list for each 
    sentence.
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    return [tokenizer.tokenize(sent)
            for sent in sent_tokenizer.tokenize(text)]
