# -*- coding: utf-8 -*-

'''
Functions to extract features from the pairs, to be used by
machine learning algorithms.
'''

from __future__ import division
import utils
import datastructures
import numpy as np

def create_lda_vectors(pairs):
    pass


def words_in_common(pair): 
    '''
    Return the proportion of words in common in a pair.
    Repeated words are ignored.
    
    :type pair: datastructures.Pair
    :return: a tuple with the proportion of common words in the first
        sentence and in the second one
    '''
    tokens_t = pair.annotated_t.tokens
    tokens_h = pair.annotated_h.tokens
    
    tokens_t = set(tokens_t)
    tokens_h = set(tokens_h)
    
    num_common_tokens = len(tokens_h.intersection(tokens_t))
    proportion_t = num_common_tokens / len(tokens_t)
    proportion_h = num_common_tokens / len(tokens_h)
    
    return (proportion_t, proportion_h)

def pipeline_minimal(pairs):
    '''
    Process the pairs and return a tuple (x, y, z) with feature representations,
    entailment classes and similarity scores.
    
    The pipeline includes the minimal preprocessing and feature extraction.
    '''
    utils.preprocess_minimal(pairs)
    x = extract_features_minimal(pairs)
    y = extract_classes(pairs)
    z = np.array([p.similarity for p in pairs])
    
    return (x, y, z)

def extract_features_minimal(pairs):
    '''
    Extract features from the given pairs to be used in a classifier.
    
    Minimalist function. It only extract the proportion of common words.
    
    :return: a numpy 2-dim array
    '''
    features = np.array([words_in_common(pair) for pair in pairs])
    return features
    
def extract_classes(pairs):
    '''
    Extract the class infomartion (paraphrase, entailment, none, contradiction)
    from the pairs. 
    
    :return: a numpy array with values from 0 to num_classes - 1
    '''
    classes = np.array([pair.entailment.value - 1 for pair in pairs])
    return classes


def negation_check(t, h):
    '''
     Check if a verb from H is negated in T. Negation is understood both as the
     verb itself negated by an adverb such as not or never, or by the presence
     of a non-negated antonym.
     :type t: datastructures.Sentence
     :type h: datastructures.Sentence
     :return: 1 for negation, 0 otherwise
    '''
    # TODO: the POS check is based on the penn treebank tagset. we could make it more general
    verb_lemmas_h = set(token.lemma
                        for token in h
                        if token.pos[0] == 'V')
    verbs_t = set(token
                  for token in t
                  if token.pos[0] == 'V' and token.lemma in verb_lemmas_h)
    
    #TODO: add synonyms to verbs from H
    for verb in verbs_t:
        # check if it is negated
        dependents = verb.dependents
        for dependent in dependents:
            if dependent.dependency_relation == 'neg':
                return 1
    
    return 0

