# -*- coding: utf-8 -*-

'''
Script to count words and verbs in RTE corpora.
'''

from __future__ import division

import argparse
import numpy as np
import logging


import utils

def compute_statistics(pairs):
    '''
    Count words and verbs in each T and H.
    '''
    n = len(pairs)
    sents_t = np.zeros(n)
    sents_h = np.zeros(n)
    tokens_t = np.zeros(n)
    tokens_h = np.zeros(n)
    verbs_t = np.zeros(n)
    verbs_h = np.zeros(n)
    
    total_sents_t = 0
    total_sents_h = 0
    total_tokens_t = 0
    total_tokens_h = 0
    total_verbs_t = 0
    total_verbs_h = 0
    
#     tagger = utils.load_senna_tagger()
    tagger = utils.load_nlpnet_tagger()
    
    for i, pair in enumerate(pairs):
        tagged_t_sents = tagger.tag(pair.t)
        tagged_h_sents = tagger.tag(pair.h)
        
        sents_t[i] = len(tagged_t_sents)
        sents_h[i] = len(tagged_h_sents)
        
        tokens_t[i] = sum(len(sent) for sent in tagged_t_sents)
        tokens_h[i] = sum(len(sent) for sent in tagged_h_sents)
        
        num_verbs_t = len([tag for sent in tagged_t_sents
                           for _, tag in sent
                           if tag[0] == 'V'])
        num_verbs_h = len([tag for sent in tagged_h_sents
                           for _, tag in sent
                           if tag[0] == 'V'])
        
        verbs_t[i] = num_verbs_t
        verbs_h[i] = num_verbs_h
#         t_sents = utils.tokenize(pair.t)
#         h_sents = utils.tokenize(pair.h)
#         
#         total_sents_t += len(t_sents)
#         total_sents_h += len(h_sents)
#         
#         for sent in t_sents:
#             verbs = [tag for _, tag in tagger.tag(sent) if tag[0] == 'V']
#             
#             total_tokens_t += len(sent)
#             total_verbs_t += len(verbs)
#         
#         for sent in h_sents:
#             verbs = [tag for _, tag in tagger.tag(sent) if tag[0] == 'V']
#             
#             total_tokens_h += len(sent)
#             total_verbs_h += len(verbs)
    
#     num_pairs = float(len(pairs))
    print 'Average followed by std deviation'
    print 'Number for T followed by numbers for H'
    print 'Number of sentences, sentence length and number of verbs'
    print '\t'.join([str(x) for x in [sents_t.mean(), sents_t.std(),
                                      sents_h.mean(), sents_h.std(),
                                      tokens_t.mean(), tokens_t.std(),
                                      tokens_h.mean(), tokens_h.std(),
                                      verbs_t.mean(), verbs_t.std(),
                                      verbs_h.mean(), verbs_h.std()]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='RTE file in XML format')
    args = parser.parse_args()
    
    logging.info('Reading XML')
    pairs = utils.read_xml(args.input)
    
    logging.info('Computing statistics')
    compute_statistics(pairs)

