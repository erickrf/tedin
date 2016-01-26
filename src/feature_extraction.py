# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Functions to extract features from the pairs, to be used by
machine learning algorithms.
'''

from __future__ import division
import numpy as np
import logging
from operator import xor

import utils
import numerals
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

def pipeline_minimal(pairs, model_config):
    '''
    Process the pairs and return a numpy array with feature representations.
    
    The pipeline includes the minimal preprocessing and feature extraction.
    '''
    utils.preprocess_minimal(pairs)
    x = extract_features_minimal(pairs, model_config)
    
    return x

def pipeline_dependency(pairs):
    '''
    Process the pairs and return a numpy array with feature representations.
    
    The pipeline follows the work of Sharma et al. (2015)
    '''
    utils.preprocess_dependency(pairs)

def load_stopwords(path=None):
    '''
    Load the stopwords from a file.
    
    :param path: the file containing stopwords. If None, the default
        from global configuration is read.
    :return type: set or None
    '''
    if path is None:
        path = config.stopwords_path
        
    if path is None or path == '':
        logging.warning('No stopword file set. Stopwords won\'t be treated.')
        return None
    
    with open(path, 'rb') as f:
        text = unicode(f.read(), 'utf-8')
    
    stopwords = set(text.splitlines())
    return stopwords

def extract_features_minimal(pairs, model_config):
    '''
    Extract features from the given pairs to be used in a classifier.
    
    Minimalist function. It only extract the proportion of common words.
    
    :return: a numpy 2-dim array
    '''
    stopwords = load_stopwords(model_config.stopwords_path)
    features = np.array([words_in_common(pair, stopwords) for pair in pairs])
    
    return features

def quantity_agreement(pair):
    '''
    Check if quantities on t and h match.
    
    Only checks quantities modifying aligned heads. This returns 0 if there is a mismatch
    and 1 otherwise.
    
    :type pair: datastructures.Pair
    '''
    for token_t, token_h in pair.lexical_alignments:
        # let's assume only one quantity modifier for each token
        # (honestly, how could there be more than one?)
        quantity_t = [d for d in token_t.dependents
                      if d.dependency_relation == 'num']
        quantity_h = [d for d in token_h.dependents
                      if d.dependency_relation == 'num']
        
        if len(quantity_t) > 1 or len(quantity_h) > 1:
            msg = 'More than one quantity modifier in "{}"'
            logging.warning(msg.format(pair.t))
        
        if len(quantity_t) == 0 or len(quantity_h) == 0:
            continue
        
        quantity_t = numerals.get_number(quantity_t[0])
        quantity_h = numerals.get_number(quantity_h[0])
        if quantity_h != quantity_t:
            msg = 'Quantities differ in pair {}: {} and {}'
            logging.debug(msg.format(pair.id, quantity_t, quantity_h))
            return 0
    
    return 1

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

def matching_verb_arguments(pair):
    '''
    Return 1 if there is a matching verb in both sentences with a matching subject
    and object.
    '''
    for token1, token2 in pair.lexical_alignments:
        # check pairs of aligned verbs
        if token1.pos != 'VERB' or token2.pos != 'VERB':
            continue
        
        # check if the arguments in H have a corresponding one in T
        subj_h = [dep for dep in token2.dependents if dep.dependency_relation == 'nsubj']
        dobj_h = [dep for dep in token2.dependents if dep.dependency_relation == 'dobj']
        adpobj_h = [dep for dep in token2.dependents if dep.dependency_relation == 'adpobj']
        
        subj_t = [dep for dep in token2.dependents if dep.dependency_relation == 'nsubj'][0]
        dobj_t = [dep for dep in token2.dependents if dep.dependency_relation == 'dobj'][0]
        adpobj_t = [dep for dep in token2.dependents if dep.dependency_relation == 'adpobj'][0]
        
        if subj_h.text != subj_t.text:
            # subjects must match exactly
            #TODO: check passives
            continue
        
        any_object_t = [dobj_t.text, adpobj_t.text]
        if dobj_h.text in any_object_t or adpobj_h in any_object_t:
            # either a match of direct object or adpositional object is fine
            return 1
    
    return 0

def dependency_overlap(pair, both):
    '''
    Check how many of the dependencies on the pairs match. Return the ratio between
    dependencies in both sentences and those only in H (or also in T if `both` 
    is True). 
    
    :type pair: datastructures.Pair
    :param both: if True, return a tuple with the ratio to dependencies in T and H.
    '''
    # dependencies are stored as a tuple of 3 string: dependency label, head
    # and modifier. This function doesn't check lemmas or anything.
    deps_t = pair.annotated_t.dependencies
    deps_h = pair.annotated_h.dependencies
    
    num_common = len(deps_t.intersection(deps_h))
    ratio_h = num_common / len(deps_h)
    
    if both:
        ratio_t = num_common / len(deps_t)
        return (ratio_t, ratio_h)
    else:
        return ratio_h

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

