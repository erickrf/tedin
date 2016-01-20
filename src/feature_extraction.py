# -*- coding: utf-8 -*-

'''
Functions to extract features from the pairs, to be used by
machine learning algorithms.
'''

from __future__ import division
import numpy as np
from operator import xor

import utils
import datastructures
import config

def create_lda_vectors(pairs):
    pass


def words_in_common(pair, stopwords=None): 
    '''
    Return the proportion of words in common in a pair.
    Repeated words are ignored.
    
    :type pair: datastructures.Pair
    :type stopwords: set
    :return: a tuple with the proportion of common words in the first
        sentence and in the second one
    '''
    tokens_t = pair.annotated_t.tokens
    tokens_h = pair.annotated_h.tokens
    
    if stopwords is None:
        stopwords = set()
    
    tokens_t = set(token for token in  tokens_t if token not in stopwords)
    tokens_h = set(token for token in  tokens_h if token not in stopwords)
    
    num_common_tokens = len(tokens_h.intersection(tokens_t))
    proportion_t = num_common_tokens / len(tokens_t)
    proportion_h = num_common_tokens / len(tokens_h)
    
    return (proportion_t, proportion_h)

def pipeline_minimal(pairs):
    '''
    Process the pairs and return a numpy array with feature representations.
    
    The pipeline includes the minimal preprocessing and feature extraction.
    '''
    utils.preprocess_minimal(pairs)
    x = extract_features_minimal(pairs)
    
    return x

def pipeline_dependency(pairs):
    '''
    Process the pairs and return a numpy array with feature representations.
    
    The pipeline follows the work of Sharma et al. (2015)
    '''
    utils.preprocess_dependency(pairs)

def load_stopwords():
    '''
    Load the stopwords from a file set in the config.
    
    :return type: set or None
    '''
    path = config.stopwords_path
    if path is None or path == '':
        return None
    
    with open(path, 'rb') as f:
        text = unicode(f.read(), 'utf-8')
    
    stopwords = set(text.splitlines())
    return stopwords

def extract_features_minimal(pairs):
    '''
    Extract features from the given pairs to be used in a classifier.
    
    Minimalist function. It only extract the proportion of common words.
    
    :return: a numpy 2-dim array
    '''
    stopwords = load_stopwords()
    features = np.array([words_in_common(pair, stopwords) for pair in pairs])
    
    return features

def is_negated(verb):
    '''
    Check if a verb is negated in the syntactic tree. This function searches its
    direct syntactic children for a negation relation. 
    
    :type verb: datastructures.Token
    :return: bool
    '''
    for dependent in verb.dependents:
        if dependent.dependency_relation == config.negation_rel:
            return True
    
    return False

def negation_check(t, h):
    '''
     Check if a verb from H is negated in T. Negation is understood both as the
     verb itself negated by an adverb such as not or never, or by the presence
     of a non-negated antonym.
     
     :type t: datastructures.Sentence
     :type h: datastructures.Sentence
     :return: 1 for negation, 0 otherwise
    '''
    # the POS check is based on the Universal Treebanks tagset
    verbs_h = [token for token in h.tokens
               if token.pos == 'VERB']
    verbs_t = [token for token in t.tokens
               if token.pos == 'VERB']
    
    for verb_t in verbs_t:
        # check if it is in H
        #TODO: also check synonyms
        # we do a linear search instead of using a set, yes
        # the overhead of creating a set to check 1-4 verbs is not worth it
        for verb_h in verbs_h:
            if verb_t.lemma == verb_h.lemma:
                t_negated = is_negated(verb_t)
                h_negated = is_negated(verb_h)
                if xor(t_negated, h_negated):
                    return 1
                
    return 0

