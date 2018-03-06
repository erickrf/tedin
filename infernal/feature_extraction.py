# -*- coding: utf-8 -*-

from __future__ import unicode_literals, division

"""
Functions to extract features from the pairs, to be used by
machine learning algorithms.
"""

import nltk
import logging
from operator import xor
import numpy as np
from scipy.spatial.distance import cosine, cdist
import zss

from . import numerals
from . import datastructures
from . import config
from . import openwordnetpt as own


def _is_negated(verb):
    """
    Check if a verb is negated in the syntactic tree. This function searches its
    direct syntactic children for a negation relation.

    :type verb: datastructures.Token
    :return: bool
    """
    for dependent in verb.dependents:
        if dependent.dependency_relation == config.negation_rel:
            return True

    return False


def _count_dependency_matches(deps1, deps2):
    """
    Checks how many of the deps1 have a matching dependency in deps2

    :param deps1: list or set of Dependency objects
    :param deps2: list or set of Dependency objects
    :return: integer
    """
    matches = 0
    for dep1 in deps1:
        for dep2 in deps2:
            if dep1.is_equivalent(dep2):
                matches += 1
                break

    return matches


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


class FeatureExtractor(object):
    """
    Class to extract features from pairs.
    """
    def __init__(self, both, stopwords=None, ed=None):
        """

        :param both: whether to extract bidrectional features when applicable.
        :param stopwords: set of stopwords, or None
        :param ed: utils.EmbeddingDictionary
        """
        self.both = both
        self.ed = ed
        if stopwords is None:
            stopwords = set()
        self.stopwords = stopwords

    def extract_features(self, pair):
        self._compute_common_values(pair)

    def _compute_common_values(self, pair):
        """
        Pre-compute some values used by different functions
        """
        tokens_t = pair.annotated_t.tokens
        tokens_h = pair.annotated_h.tokens

        self.distances = None
        self.content_tokens_t = [token for token in tokens_t
                                 if token.lemma not in self.stopwords]
        self.content_lemma_set_t = set(token.lemma
                                       for token in self.content_tokens_t)
        self.content_tokens_h = [token for token in tokens_h
                                 if token.lemma not in self.stopwords]
        self.content_lemma_set_h = set(token.lemma
                                       for token in self.content_tokens_h)

        self.embeddings_t = np.array([self.ed[token.text.lower()]
                                      for token in tokens_t])
        self.embeddings_h = np.array([self.ed[token.text.lower()]
                                      for token in tokens_h])
        self.cosine_distances = cdist(self.embeddings_t, self.embeddings_h,
                                      'cosine')

    def word_overlap_proportion(self):
        """
        Return the proportion of words in common appearing in H.

        :return: the proportion of words in H that also appear in T
        """
        # TODO: sets instead of lists?
        num_lemmas_t = len(self.content_lemma_set_t)
        num_lemmas_h = len(self.content_lemma_set_h)

        intersection = self.content_lemma_set_h.intersection(
            self.content_lemma_set_t)
        num_common_tokens = len(intersection)
        proportion_t = num_common_tokens / num_lemmas_t
        proportion_h = num_common_tokens / num_lemmas_h

        return proportion_t, proportion_h

    def cosine_tree_distance(self, pair, return_operations=False):
        """
        Compute the tree edit distance of the sentences considering the
        replacement cost of two words as their embeddings's cosine distance.

        :return: an integer
        """
        def get_children(node):
            return node.dependents

        def insert_cost(node):
            return 1

        def remove_cost(node):
            return 1

        #TODO: different update costs for stopwords?
        def update_cost(node1, node2):
            # -1 because conll id's start from 1
            i = node1.id - 1
            j = node2.id - 1
            return self.cosine_distances[i, j]

        tree_t = pair.annotated_t
        tree_h = pair.annotated_h
        root_t = tree_t.root
        root_h = tree_h.root

        return zss.distance(root_t, root_h, get_children, insert_cost,
                            remove_cost, update_cost,
                            return_operations=return_operations)

    def simple_tree_distance(self, pair, return_operations=False):
        """
        Compute a simple tree edit distance (TED) value and operations.

        Nodes are considered to match if they have the same dependency label and
        lemma.
        """
        def get_children(node):
            return node.dependents

        def get_label(node):
            return node.lemma, node.dependency_relation

        def label_dist(label1, label2):
            return int(label1 != label2)

        tree_t = pair.annotated_t
        tree_h = pair.annotated_h
        root_t = tree_t.root
        root_h = tree_h.root

        return zss.simple_distance(root_t, root_h, get_children,
                                   get_label, label_dist,
                                   return_operations=return_operations)

    def word_synonym_overlap_proportion(self, pair):
        """
        Like `word_overlap_proportion` but count wordnet synonyms
        as matches
        """
        alignments = [(token1, token2)
                      for token1, token2 in pair.lexical_alignments
                      if token1.lemma not in self.stopwords
                      and token2.lemma not in self.stopwords]

        num_common_tokens = len(alignments)
        num_lemmas_t = len(self.content_lemma_set_t)
        num_lemmas_h = len(self.content_lemma_set_h)

        proportion_t = num_common_tokens / num_lemmas_t
        proportion_h = num_common_tokens / num_lemmas_h

        return proportion_t, proportion_h

    def soft_word_overlap(self):
        """
        Compute the "soft" word overlap between the sentences. It is
        defined as the average of the maximum embedding similarity of
        each word in one sentence in relation to the other.

        sum_i max_similarity(t_i, h)) / len(t)

        where max_similarity returns the highest similarity value (cosine)
        of words in H with respect to t_i.

        :return: return a tuple (similarity1, similarity2)
        """
        content_token_ids_t = [token.id - 1 for token in self.content_tokens_t]
        content_token_ids_h = [token.id - 1 for token in self.content_tokens_h]

        dists = self.cosine_distances[content_token_ids_t, content_token_ids_h]

        # dists has shape (num_tokens1, num_tokens2)
        # min_dists1 has the minimum distance from each word in T to any in H
        # min_dists2 is the opposite

        # if T has a word without anything similar in H, similarities1 will
        # decrease. And if all words in H have a similar one in T, similarities2
        # will be high
        min_dists2 = dists.min(0)
        similarities2 = 1 - min_dists2
        mean2 = similarities2.mean()

        if self.both:
            min_dists1 = dists.min(1)
            similarities1 = 1 - min_dists1
            mean1 = similarities1.mean()

            return mean1, mean2

        return mean2

    def sentence_cosine(self):
        """
        Compute the cosine of the mean embedding vector of the two sentences.

        :return: an integer, the cosine similarity
        """
        avg1 = self.embeddings_t.mean(0)
        avg2 = self.embeddings_h.mean(0)

        # cosine returns the cosine distance:
        # dist(x, y) = 1 - cos(x, y) --> cos(x, y) = 1 - dist(x, y)
        dist = cosine(avg1, avg2)
        cos_similarity = 1 - dist

        return cos_similarity

    def quantity_agreement(self, pair):
        """
        Check if quantities on t and h match.

        Only checks quantities modifying aligned heads. This returns 0 if there
        is a mismatch and 1 otherwise.

        :type pair: datastructures.Pair
        """
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

    #TODO: this feature is weird. does it help at all?
    def matching_verb_arguments(self, pair):
        """
        Check if there is at least one verb matching in the two sentences and, if
        so, its object and subject are the same. In case of a positive result,
        return 1.

        :param both: if True, return a tuple with two values. The first is 1
        if T and H have the same subject, T has an object and H not. The second
        is 1 if both have the same subject, H has an object and T not.
        """
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
                return (1, 1) if self.both else 1

            if self.both:
                if dobj_t is None and dobj_h is not None:
                    at_least = (0, 1)
                if dobj_t is not None and dobj_h is None:
                    at_least = (1, 0)

            if dobj_t is not None and dobj_h is not None:
                if dobj_t.lemma == dobj_h.lemma:
                    return (1, 1) if self.both else 1

        if at_least:
            return at_least

        return (0, 0) if self.both else 0

    def dependency_overlap(self, pair):
        """
        Check how many of the dependencies on the pairs match. Return the ratio
        between dependencies in both sentences and those only in H (or also in T
         if `both` is True).

        :type pair: datastructures.Pair
        """
        # dependencies are stored as a tuple of dependency label, head
        # and modifier.
        deps_t = set(pair.annotated_t.dependencies)
        deps_h = set(pair.annotated_h.dependencies)

        # matches_t and matches_h can be different
        matches_h = _count_dependency_matches(deps_h, deps_t)
        ratio_h = matches_h / len(deps_h)
        if self.both:
            matches_t = _count_dependency_matches(deps_t, deps_h)
            ratio_t = matches_t / len(deps_t)

            return ratio_t, ratio_h

        return ratio_h

    def has_nominalization(self, pair):
        """
        Check whether a verb in T has a corresponding nominalization in H.
        If so, return 1.

        :type pair: datastructures.Pair
        :param both: if True, also check verbs in H with nouns in T.
        """
        sent1 = pair.annotated_t
        sent2 = pair.annotated_h
        val1 = _has_nominalization(sent1, sent2)

        if self.both:
            val2 = _has_nominalization(sent2, sent1)
            return val1, val2
        else:
            return val1

    def bleu(self, pair):
        """
        Return the BLEU score from the first sentence to the second.
        If `both` is True, also return the opposite.
        """
        tokens1 = [t.lemma for t in pair.annotated_t.tokens]
        tokens2 = [t.lemma for t in pair.annotated_h.tokens]

        bleu1 = nltk.translate.bleu([tokens1], tokens2)
        if self.both:
            bleu2 = nltk.translate.bleu([tokens2], tokens1)
            return bleu1, bleu2

        return bleu1

    def length_proportion(self):
        """
        Compute the proportion of the size of T to H.

        If stopwords are given, they are removed prior the computation.
        """
        length_t = len(self.content_tokens_t)
        length_h = len(self.content_tokens_h)
        return length_t / length_h

    def negation_check(self, pair):
        """
         Check if a verb from H is negated in T. Negation is understood both as the
         verb itself negated by an adverb such as not or never, or by the presence
         of a non-negated antonym.

         :type pair: datastructures.Pair
         :return: 1 for negation, 0 otherwise
        """
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
                if own.are_synonyms(verb_t.lemma, verb_h.lemma):
                    t_negated = _is_negated(verb_t)
                    h_negated = _is_negated(verb_h)
                    if xor(t_negated, h_negated):
                        return 1

        return 0

