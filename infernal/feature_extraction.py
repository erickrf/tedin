# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division

'''
Functions to extract features from the pairs, to be used by
machine learning algorithms.
'''

import nltk
import logging
from operator import xor
import numpy as np
from scipy.spatial.distance import cdist
import zss

from . import numerals
from . import datastructures
from . import config
from . import openwordnetpt as own


def word_overlap_proportion(pair, stopwords=None):
    '''
    Return the proportion of words in common appearing in H.

    :type pair: datastructures.Pair
    :type stopwords: set
    :return: the proportion of words in H that also appear in T
    '''
    tokens_t = pair.annotated_t.tokens
    tokens_h = pair.annotated_h.tokens

    if stopwords is None:
        stopwords = set()

    tokens_t = set(token.lemma
                   for token in tokens_t if token.lemma not in stopwords)
    tokens_h = set(token.lemma
                   for token in tokens_h if token.lemma not in stopwords)

    num_common_tokens = len(tokens_h.intersection(tokens_t))
    proportion_t = num_common_tokens / len(tokens_t)
    proportion_h = num_common_tokens / len(tokens_h)

    return proportion_t, proportion_h


def simple_tree_distance(pair, return_operations=False):
    '''
    Extract a simple tree edit distance (TED) value and operations.

    Nodes are considered to match if they have the same dependency label and
    lemma.
    '''
    def get_children(node):
        return node.dependents

    def get_label(node):
        return node.index
        # return node.lemma, node.dependency_relation

    def label_dist(label1, label2):
        return int(label1 != label2)

    tree_t = pair.annotated_t
    tree_h = pair.annotated_h
    root_t = tree_t.root
    root_h = tree_h.root

    return zss.simple_distance(root_t, root_h, get_children,
                               get_label, label_dist,
                               return_operations=return_operations)


def word_synonym_overlap_proportion(pair, stopwords=None):
    '''
    Like `word_overlap_proportion` but count wordnet synonyms
    as matches
    '''
    if stopwords is None:
        stopwords = set()

    alignments = [(token1, token2)
                  for token1, token2 in pair.lexical_alignments
                  if token1.lemma not in stopwords
                  and token2.lemma not in stopwords]

    num_common_tokens = len(alignments)
    num_tokens_t = len(set(t for t in pair.annotated_t.tokens
                           if t.lemma not in stopwords))
    num_tokens_h = len(set(t for t in pair.annotated_h.tokens
                           if t.lemma not in stopwords))

    proportion_t = num_common_tokens / num_tokens_t
    proportion_h = num_common_tokens / num_tokens_h

    return proportion_t, proportion_h


def soft_word_overlap(pair, embeddings):
    '''
    Compute the "soft" word overlap between the sentences. It is
    defined as the average of the maximum embedding similarity of
    each word in one sentence in relation to the other.

    sum_i max_similarity(t_i, h)) / len(t)

    where max_similarity returns the highest similarity value (cosine)
    of words in H with respect to t_i.

    :param pair: a Pair object
    :param embeddings: a utils.EmbeddingDictionary
    :return: return a tuple (similarity1, similarity2)
    '''
    embeddings1 = np.array([embeddings[token]
                            for token in pair.annotated_t.lower_content_tokens])
    embeddings2 = np.array([embeddings[token]
                            for token in pair.annotated_h.lower_content_tokens])
    dists = cdist(embeddings1, embeddings2, 'cosine')

    # dists has shape (num_tokens1, num_tokens2)
    # min_dists1 has the minimum distance from each word in T to any in H
    # min_dists2 is the opposite
    min_dists1 = dists.min(1)
    min_dists2 = dists.min(0)

    # if T has a word without anything similar in H, similarities1 will
    # decrease. And if all words in H have a similar one in T, similarities2
    # will be high
    similarities1 = 1 - min_dists1
    similarities2 = 1 - min_dists2

    mean1 = similarities1.mean()
    mean2 = similarities2.mean()
    return mean1, mean2


def sentence_average_embeddings(pair, embeddings):
    '''
    Concatenate embeddings of both sentences. Each sentence embedding is obtained
    as the average of their words.

    :param pair: ds.Pair
    :type embeddings: utils.EmbeddingDictionary
    '''
    embeddings1 = embeddings.get_sentence_embeddings(pair.annotated_t)
    embeddings2 = embeddings.get_sentence_embeddings(pair.annotated_h)
    avg1 = embeddings1.mean(0)
    avg2 = embeddings2.mean(0)

    return np.concatenate((avg1, avg2))


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


#TODO: this feature is weird. does it help at all?
def matching_verb_arguments(pair, both=True):
    '''
    Check if there is at least one verb matching in the two sentences and, if so,
    its object and subject are the same. In case of a positive result, return 1.

    :param both: if True, return a tuple with two values. The first is 1
    if T and H have the same subject, T has an object and H not. The second
    is 1 if both have the same subject, H has an object and T not.
    '''
    at_least = None

    for token1, token2 in pair.lexical_alignments:

        # workaround... in openwordnet-pt, "ser" and "ter" share one
        # synset in which both words don't really mean the exact same thing.
        # This can lead to many false positives due to their frequencies
        if (token1.lemma == 'ser' and token2.lemma == 'ter') or \
        (token1.lemma == 'ter' and token2.lemma == 'ser'):
            continue

        # check pairs of aligned verbs
        if token1.pos != 'VERB' or token2.pos != 'VERB':
            continue

        # check if the arguments in H have a corresponding one in T
        # due to parser errors, we could have more than one subject or
        # dobj per verb, but here we take only the first occurrence.
        subj_t = token1.get_dependent('nsubj')
        subj_h = token2.get_dependent('nsubj')
        if subj_h is not None and subj_t is not None:
            if subj_h.lemma != subj_t.lemma:
                continue

        dobj_t = token1.get_dependent('dobj')
        dobj_h = token2.get_dependent('dobj')
        if dobj_h is None and dobj_t is None:
            return (1, 1) if both else 1

        if both:
            if dobj_t is None and dobj_h is not None:
                at_least = (0, 1)
            if dobj_t is not None and dobj_h is None:
                at_least = (1, 0)

        if dobj_t is not None and dobj_h is not None:
            if dobj_t.lemma == dobj_h.lemma:
                return (1, 1) if both else 1

    if at_least:
        return at_least

    return (0, 0) if both else 0


def dependency_overlap(pair, both=False):
    '''
    Check how many of the dependencies on the pairs match. Return the ratio between
    dependencies in both sentences and those only in H (or also in T if `both`
    is True).

    :type pair: datastructures.Pair
    :param both: if True, return a tuple with the ratio to dependencies in T and H.
    '''
    # dependencies are stored as a tuple of dependency label, head
    # and modifier.
    deps_t = set(pair.annotated_t.dependencies)
    deps_h = set(pair.annotated_h.dependencies)

    num_common = len(deps_t.intersection(deps_h))
    ratio_h = num_common / len(deps_h)

    if both:
        ratio_t = num_common / len(deps_t)
        return (ratio_t, ratio_h)
    else:
        return ratio_h


def _has_nominalization(sent1, sent2):
    """
    Internal helper function. Returns 1 if a verb in sent 1 has a nominalization
    in sent2, 0 otherwise.
    :param sent1: datastructures.Sentence
    :param sent2: datastructures.Sentence
    """
    verbs1 = [token.lemma for token in sent1.tokens if token.pos == 'VERB']
    # lemmas2 = set(token.lemma for token in sent2.tokens)
    for verb in verbs1:
        nominalizations = own.find_nominalizations(verb)
        for nominalization in nominalizations:
            for token2 in sent2.tokens:
                if token2.lemma == nominalization and \
                                token2.dependency_relation == 'dobj':
                    return 1

    return 0


def has_nominalization(pair, both=False):
    """
    Check whether a verb in T has a corresponding nominalization in H.
    If so, return 1.

    :type pair: datastructures.Pair
    :param both: if True, also check verbs in H with nouns in T.
    """
    own.load_wordnet(config.ownpt_path)
    sent1 = pair.annotated_t
    sent2 = pair.annotated_h
    val1 = _has_nominalization(sent1, sent2)

    if both:
        val2 = _has_nominalization(sent2, sent1)
        return val1, val2
    else:
        return val1


def bleu(pair, both):
    """
    Return the BLEU score from the first sentence to the second.
    If `both` is True, also return the opposite.
    """
    tokens1 = [t.lemma for t in pair.annotated_t.tokens]
    tokens2 = [t.lemma for t in pair.annotated_h.tokens]

    bleu1 = nltk.translate.bleu([tokens1], tokens2)
    if both:
        bleu2 = nltk.translate.bleu([tokens2], tokens1)
        return bleu1, bleu2

    return bleu1


def length_proportion(pair):
    """
    Compute the proportion of the size of T to H.

    If stopwords are given, they are removed prior the computation.
    """
    length_t = len(pair.annotated_t.lower_content_tokens)
    length_h = len(pair.annotated_h.lower_content_tokens)
    return length_t / length_h


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

