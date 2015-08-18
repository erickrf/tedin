# -*- coding: utf-8 -*-

'''
Utility functions
'''

import subprocess
import os
from xml.etree import cElementTree as ET
import nltk
import nlpnet

import config
import datastructures

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
            entailment = attribs['entailment'].lower() in ['yes', 'entailment']
            
            # this is deleted because the Pair object has entailment as an explict argument
            del attribs['entailment']
            
        elif 'value' in attribs:
            entailment = attribs['value'].lower() == 'true'
            
            # same as above
            del attribs['value']
        
        pair = datastructures.Pair(t, h, entailment, **attribs)
        pairs.append(pair)
    
    return pairs


def call_corenlp(text):
    cp = '"%s"' % os.path.join(config.corenlp_path, '*')
    args = ['java', '-mx3g', '-cp', cp, 
            'edu.stanford.nlp.pipeline.StanfordCoreNLP',
            '-annotators', 'tokenize,ssplit,pos,lemma,depparse']
    
    # in windows, subprocess call only works using a string instead of a list
    cmd = ' '.join(args)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
    stdout = p.communicate(text)[0]
    return stdout


def preprocess(pairs):
    '''
    Preprocess the given pairs to add linguistic knowledge.
    '''
    for i, pair in enumerate(pairs):
        pair.annotated_t = call_corenlp(pair.t)
        pair.annotated_h = call_corenlp(pair.h)
        pairs[i] = pair


def load_senna_tagger():
    '''
    Load and return the SENNA tagger.
    Its location in the file system is read from the config module.
    '''
    return nltk.tag.SennaTagger(config.senna_path)

def load_nlpnet_tagger():
    '''
    Load and return the nlpnet POS tagger.
    Its location in the file system is read from the config module.
    '''
    return nlpnet.POSTagger(config.nlpnet_path)

def tokenize(text):
    """
    Return a list of lists of the tokens in the text. One list for each 
    sentence.
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    return [tokenizer.tokenize(sent)
            for sent in sent_tokenizer.tokenize(text)]
