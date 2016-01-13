# -*- coding: utf-8 -*-

'''
Functions for calling external NLP tools.
'''

import urllib
import subprocess
import os
import nlpnet
import nltk
import tempfile

import config

nlpnet_tagger = None

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
    # store in global variable to avoid loading times
    global nlpnet_tagger
    if nlpnet_tagger is not None:
        return nlpnet_tagger
    
    nlpnet_tagger = nlpnet.POSTagger(config.nlpnet_path_pt)
    return nlpnet_tagger

def call_malt(text):
    '''
    Call the MALT parser locally and return a parsed sentence.
    It uses nlpnet internally to POS tag the text.
    
    :param text: a single sentence
    '''
    nlpnet_tagger = load_nlpnet_tagger()
    tagged = nlpnet_tagger.tag(text)[0]
    
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
            
    input_file.write(b'\n\n')
    input_file.close()
    
    cmd = ['java' ,
           '-jar', 'maltparser-1.8.1/maltparser-1.8.1.jar',
           '-c'  , 'uni-dep-tb-ptbr', 
           '-i'  , input_file.name,
           '-o'  , output_file.name, 
           '-m'  , 'parse']

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = p.wait()
    output_file.close()
    
    if ret != 0:
        print 'Return code from MALT parser not 0!'
        print ret
    
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

