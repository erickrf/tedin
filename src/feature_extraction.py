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
import resources

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
    '''
    utils.preprocess_dependency(pairs)
    
    extractors = [dependency_overlap, negation_check, quantity_agreement]
    
    # some feature extractors return tuples, others return ints
    # convert each one to numpy and then join them
    feature_arrays = [[np.array(f(pair)) for f in extractors]
                      for pair in pairs]


def extract_features_minimal(pairs, model_config):
    '''
    Extract features from the given pairs to be used in a classifier.
    
    Minimalist function. It only extract the proportion of common words.
    
    :return: a numpy 2-dim array
    '''
    stopwords = resources.load_stopwords(model_config.stopwords_path)
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

def _is_negated(verb):
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

# #TODO: this feature is weird. does it help at all?
# def matching_verb_arguments(pair, both=True):
#     '''
#     Check if there is at least one verb matching in the two sentences and, if so,
#     its object and objects are the same. In case of a positive result, return 1.
#     If one sentence contains an argument (subject or object) but the other 
#     doesn't, it also results in 1.
#     
#     :param both: if True, return a tuple with two values. The first considers 
#     whether T->H and the second, H->T. In other words, if H has an object absent
#     in T, it would return (0, 1), provided the rest of the verb structure matches.
#     '''
#     def match():
#         pass
#     
#     for token1, token2 in pair.lexical_alignments:
#         # check pairs of aligned verbs
#         if token1.pos != 'VERB' or token2.pos != 'VERB':
#             continue
#         
#         # check if the arguments in H have a corresponding one in T
#         subj_h = token2.get_dependents('nsubj')
#         if subj_h is not None:
#             # subjects must match exactly
#             #TODO: check passives
#             subj_t = token1.get_dependents('nsubj')
#             if subj_h.text != subj_t.text:
#                 return 0
#         
#         dobj_h = token2.get_dependents('dobj')
#         adpobj_h = token2.get_dependents('adpobj')
#         
#         dobj_t = token1.get_dependents('dobj')
#         adpobj_h = token1.get_dependents('adpobj')
#         
#     return 0

def dependency_overlap(pair, both=True):
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

def negation_check(pair):
    '''
     Check if a verb from H is negated in T. Negation is understood both as the
     verb itself negated by an adverb such as not or never, or by the presence
     of a non-negated antonym.
     
     :type pair: datastructures.Pair
     :return: 1 for negation, 0 otherwise
    '''
    t = pair.annotated_t
    h = pair.annotated_h
    
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
                t_negated = _is_negated(verb_t)
                h_negated = _is_negated(verb_h)
                if xor(t_negated, h_negated):
                    return 1
                
    return 0

