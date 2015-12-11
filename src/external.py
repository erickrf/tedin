# -*- coding: utf-8 -*-

'''
Functions for calling external NLP tools.
'''

import urllib
import subprocess
import os
import nlpnet
import nltk

import config

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

