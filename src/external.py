# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Functions for calling external NLP tools.
'''

import urllib
import subprocess
import nlpnet
import nltk
import tempfile
import json

import config
import utils

nlpnet_tagger = None

def load_senna_tagger():
    '''
    Load and return the SENNA tagger.
    Its location in the file system is read from the config module.
    '''
    return nltk.tag.SennaTagger(config.senna_path)

def load_nlpnet_tagger(language='pt'):
    '''
    Load and return the nlpnet POS tagger.
    Its location in the file system is read from the config module.
    '''
    # store in global variable to avoid loading times
    global nlpnet_tagger
    if nlpnet_tagger is not None:
        return nlpnet_tagger
    
    nlpnet_tagger = nlpnet.POSTagger(config.nlpnet_path_pt, language=language)
    return nlpnet_tagger

def call_malt(text):
    '''
    Call the MALT parser locally and return a parsed sentence.
    It uses nlpnet internally to POS tag the text.
    
    :param text: a single sentence
    '''
    nlpnet_tagger = load_nlpnet_tagger()
    tokens = utils.tokenize_sentence(text)
    tagged = nlpnet_tagger.tag_tokens(tokens, return_tokens=True)
    
    # temp files for malt parser input and output
    input_file = tempfile.NamedTemporaryFile(prefix='malt_input.conll',
                                             dir=tempfile.gettempdir(),
                                             delete=False)
    output_file = tempfile.NamedTemporaryFile(prefix='malt_output.conll',
                                              dir=tempfile.gettempdir(),
                                              delete=False)
    
    for (i, (word, tag)) in enumerate(tagged, start=1):
        input_str = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %\
            (i, word, '_', tag, tag, '_', '0', 'a', '_', '_')
        input_file.write(input_str.encode("utf8"))
            
    input_file.write(b'\n')
    input_file.close()
    
    cmd = ['java' ,
           '-jar', config.malt_jar,
           '-w'  , config.malt_dir,
           '-c'  , config.malt_model, 
           '-i'  , input_file.name,
           '-o'  , output_file.name, 
           '-m'  , 'parse']

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    output_file.close()
    
    ret_code = p.returncode
    if ret_code != 0:
        print 'Return code from MALT parser not 0! Error code {}'.format(ret_code)
        print stdout
        print stderr
    
    with open(output_file.name, 'rb') as f:
        output = unicode(f.read(), 'utf-8')
    
    return output
    
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
    '''
    Call Stanford corenlp, which should be running at the address specified in 
    the config module. The text should already be encoded.
    
    Only a dependency parser and POS tagger are run.
    '''
    properties = {'tokenize.whitespace': 'true', 
                  'annotators': 'tokenize,ssplit,pos,depparse',
                  'depparse.model': config.corenlp_depparse_path,
                  'pos.model': config.corenlp_pos_path,
                  'outputFormat': 'conllu'}
    
    # use json dumps function to convert the nested dictionary to a string
    properties_val = json.dumps(properties)
    params = {'properties': properties_val}
    
    encoded_params = urllib.urlencode(params)
    url = '{url}:{port}/?{params}'.format(url=config.corenlp_url, 
                                         port=config.corenlp_port, 
                                         params=encoded_params)
    
    # TODO: apparently, this is a bug in the CoreNLP server on handling UTF-8 data. 
    # See https://github.com/stanfordnlp/CoreNLP/issues/125
    # if the bug is fixed, this should be changed to utf-8  
    response = urllib.urlopen(url, text.encode('latin1'))
    output = response.read()
    
    return unicode(output, 'latin1')
