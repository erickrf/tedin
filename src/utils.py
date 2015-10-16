# -*- coding: utf-8 -*-

'''
Utility functions
'''

import subprocess
import os
import re
from xml.etree import cElementTree as ET
from nltk.tokenize import RegexpTokenizer
from xml.dom import minidom
import urllib
import nltk
import nlpnet

import config
import datastructures


def tokenize_sentence(text, preprocess=True):
    '''
    Tokenize the given sentence and applies preprocessing if requested 
    (conversion to lower case and digit substitution).
    '''
    if preprocess:
        text = re.sub(r'\d', '9', text.lower())
    
    tokenizer_regexp = ur'''(?ux)
    ([^\W\d_]\.)+|                # one letter abbreviations, e.g. E.U.A.
    \d{1,3}(\.\d{3})*(,\d+)|      # numbers in format 999.999.999,99999
    \d{1,3}(,\d{3})*(\.\d+)|      # numbers in format 999,999,999.99999
    \d+:\d+|                      # time and proportions
    \d+([-\\/]\d+)*|              # dates. 12/03/2012 12-03-2012
    [DSds][Rr][Aa]?\.|            # common abbreviations such as dr., sr., sra., dra.
    [Mm]\.?[Ss][Cc]\.?|           # M.Sc. with or without capitalization and dots
    [Pp][Hh]\.?[Dd]\.?|           # Same for Ph.D.
    [^\W\d_]{1,2}\$|              # currency
    (?:(?<=\s)|^)[\#@]\w*[A-Za-z_]+\w*|  # Hashtags and twitter user names
    -[^\W\d_]+|                   # clitic pronouns with leading hyphen
    \w+([-']\w+)*|                # words with hyphens or apostrophes, e.g. não-verbal, McDonald's
    -+|                           # any sequence of dashes
    \.{3,}|                       # ellipsis or sequences of dots
    \S                            # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)
    
    return tokenizer.tokenize(text)


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


def call_palavras(text):
    '''
    Call a webservice to run the parser Palavras
    
    :param text: the text to be parsed, in unicode.
    :return: the response string from Palavras
    '''
    params = {'sentence': text.encode('utf-8')}
    data = urllib.urlencode(params)
    f = urllib.urlopen(config.palavras_endpoint, data)
    response = f.read()
    f.close()
    
    return response

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
    return nlpnet.POSTagger(config.nlpnet_path_pt)

def tokenize(text):
    """
    Return a list of lists of the tokens in the text. One list for each 
    sentence.
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    return [tokenizer.tokenize(sent)
            for sent in sent_tokenizer.tokenize(text)]
