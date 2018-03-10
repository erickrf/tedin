# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
This module contains data structures used by the related scripts.
"""

import six
from enum import Enum
import numpy as np
from collections import Counter

from infernal import lemmatization
from infernal import openwordnetpt as own
from infernal import ppdb


content_word_tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PNOUN'}


def _compat_repr(repr_string, encoding='utf-8'):
    """
    Function to provide compatibility with Python 2 and 3 with the __repr__
    function. In Python 2, return a encoded version. In Python 3, return
    a unicode object. 
    """
    if six.PY2:
        return repr_string.encode(encoding)
    else:
        return repr_string


def filter_words_by_pos(tokens, tags=None):
    """
    Filter out words based on their POS tags.

    If no set of tags is provided, a default of content tags is used:
    {'NOUN', 'VERB', 'ADJ', 'ADV', 'PNOUN'}

    :param tokens: list of datastructures.Token objects
    :param tags: optional set of allowed tags
    :return: list of the tokens having the allowed tokens
    """
    if tags is None:
        tags = content_word_tags

    return [token for token in tokens if token.pos in tags]


class Dataset(object):
    """
    Class for storing data in the format used by TEDIN models.
    """

    def __init__(self, pairs, nodes1, nodes2, weights=None, labels=None,
                 use_weights=False):
        """
        Create a Dataset

        :param pairs: list of Pair objects
        :param nodes1: numpy array (num_pairs, max_len, num_inds) where num_inds
            is the number of indices to represent a node. Usually it is 2:
            word embedding and dependency label embedding.
        :param nodes2: numpy array (num_pairs, max_len, num_inds)
        :param labels: numpy array (num_pairs) or None (this is the target
            class)
        :param weights: the weight for each instance
        :param use_weights: compute weights for each instance inversely
            proportional to the frequency of its class
        """
        self.pairs = pairs
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.labels = labels
        self.last_batch_index = 0
        self.epoch = 1

        if weights is not None:
            self.weights = weights
        elif use_weights:
            weight_dict = self._compute_class_weights()
            self.weights = np.array([weight_dict[label]
                                     for label in labels], np.float32)
        else:
            self.weights = np.ones_like(labels, np.float32)

        self._post_assignments()

    def _compute_class_weights(self):
        """
        Compute the class weights inversally proportional to its frequency

        :return: a dictionary mapping each class to a float weight
        """
        c = Counter(self.labels)
        max_freq = max(c.values())
        weights = {class_: max_freq / c[class_] for class_ in c}

        return weights

    def reset_batch_counter(self):
        """
        Reset the batch counter in the dataset.

        New calls to next_batch will yield data from the beginning of the
        dataset.
        """
        self.last_batch_index = 0

    def shuffle(self):
        """
        Shuffle the data in the Dataset
        """
        state = np.random.get_state()
        for array in self._ordered_variables:
            # don't set the state after shuffling; we don't want the next call
            # after this function is finished to repeat the ordering
            np.random.set_state(state)
            np.random.shuffle(array)

    def _post_assignments(self):
        """
        Assign variables to keep track of stuff.
        """
        self.num_items = len(self.nodes1)

        # variables in the order they are given in the constructor
        self._ordered_variables = [self.pairs, self.nodes1,
                                   self.nodes2, self.weights]
        if self.labels is not None:
            self._ordered_variables.append(self.labels)

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        arrays = [a[item] for a in self._ordered_variables]
        return Dataset(*arrays)

    def combine(self, dataset):
        """
        Add the contents of `dataset` to this one.
        """
        self.pairs.extend(dataset.pairs)
        self.nodes1 = np.concatenate([self.nodes1, dataset.nodes1])
        self.nodes2 = np.concatenate([self.nodes2, dataset.nodes2])
        self.weights = np.concatenate([self.weights, dataset.weights])
        if self.labels is not None:
            self.labels = np.concatenate([self.labels, dataset.labels])
        self._post_assignments()

    def next_batch(self, batch_size, wrap=True, shuffle=True):
        """
        Return the next batch. If the end of the dataset is reached, it
        automatically takes element from the beginning.

        :param batch_size: int
        :param wrap: if True, wraps around the data, such that the desired batch
            size is always returned. If False, the returned data size may be
            less than `batch_size` if the end of the dataset is reached.
        :param shuffle: if wrap is True, shuffle the dataset before wrapping
        :return: Dataset
        """
        next_index = self.last_batch_index + batch_size
        batch = self[self.last_batch_index:next_index]

        if wrap and next_index > self.num_items:
            diff = batch_size - len(batch)
            if shuffle:
                self.shuffle()
            wrapped_batch = self[:diff]
            batch.combine(wrapped_batch)
            self.last_batch_index = diff
            self.epoch += 1
        else:
            self.last_batch_index = next_index

        return batch


# define an enum with possible entailment values
class Entailment(Enum):
    none = 1
    entailment = 2
    paraphrase = 3
    contradiction = 4


class Pair(object):
    """
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    """
    def __init__(self, t, h, id_, entailment, similarity=None):
        """
        :param t: the first sentence as a string
        :param h: the second sentence as a string
        :param id_: the id in the dataset. not very important
        :param entailment: instance of the Entailment enum
        :param similarity: similarity score as a float
        """
        self.t = t
        self.h = h
        self.id = id_
        self.lexical_alignments = None
        self.ppdb_alignments = None
        self.entailment = entailment
        self.annotated_h = None
        self.annotated_t = None

        if similarity is not None:
            self.similarity = similarity

    def inverted_pair(self):
        """
        Return an inverted version of this pair; i.e., exchange the
        first and second sentence, as well as the associated information.
        """
        if self.entailment == Entailment.paraphrase:
            entailment_value = Entailment.paraphrase
        else:
            entailment_value = Entailment.none

        p = Pair(self.h, self.t, self.id, entailment_value, self.similarity)
        p.annotated_t = self.annotated_h
        p.annotated_h = self.annotated_t
        return p

    def find_ppdb_alignments(self, max_length):
        """
        Find lexical and phrasal alignments in the pair according to
        transformation rules from the paraphrase database.

        This function should only be called after annotated_t and annotated_h
        have been provided.

        :param max_length: maximum length of the left-hand side (in number of
            tokens)
        """
        tokens_t = self.annotated_t.tokens
        tokens_h = self.annotated_h.tokens
        token_texts_t = [token.text.lower() for token in tokens_t]
        token_texts_h = [token.text.lower() for token in tokens_h]
        alignments = []

        # for purposes of this function, treat pronouns as content words
        global content_word_tags

        for i, token in enumerate(tokens_t):
            # check the maximum length that makes sense to search for
            # (i.e., so it doesn't go past sentence end)
            max_possible_length = min(len(tokens_t) - i, max_length)
            for length in range(1, max_possible_length):
                if length == 1 and token.pos not in content_word_tags:
                    continue

                lhs = [token for token in token_texts_t[i:i + length]]
                rhs_rules = ppdb.get_rhs(lhs)
                if not rhs_rules:
                    continue

                # now get the token objects, instead of just their text
                lhs = tokens_t[i:i + length]

                for rule in rhs_rules:
                    index = ppdb.search(token_texts_h, rule)
                    if index == -1:
                        continue
                    alignment = (lhs, tokens_h[index:index + len(rule)])
                    alignments.append(alignment)

        self.ppdb_alignments = alignments

    def find_lexical_alignments(self):
        '''
        Find the lexical alignments in the pair.

        Lexical alignments are simply two equal or synonym words.

        :return: list with the (Token, Token) aligned tuples
        '''
        # pronouns aren't content words, but let's pretend they are
        content_word_tags = {'NOUN', 'VERB', 'PRON', 'ADJ', 'ADV', 'PNOUN'}
        content_words_t = [
            token for token in filter_words_by_pos(
                self.annotated_t.tokens, content_word_tags)
            # own-pt lists ser and ter as synonyms
            if token.lemma not in ['ser', 'ter']]

        content_words_h = [
            token for token in filter_words_by_pos(
                self.annotated_h.tokens, content_word_tags)
            if token.lemma not in ['ser', 'ter']]

        lexical_alignments = []

        for token_t in content_words_t:
            nominalizations_t = own.find_nominalizations(token_t.lemma)

            for token_h in content_words_h:
                aligned = False
                if token_t.lemma == token_h.lemma:
                    aligned = True
                elif own.are_synonyms(token_t.lemma, token_h.lemma):
                    aligned = True
                elif token_h.lemma in nominalizations_t:
                    aligned = True
                elif token_t.lemma in own.find_nominalizations(token_h.lemma):
                    aligned = True

                if aligned:
                    lexical_alignments.append((token_t, token_h))

        self.lexical_alignments = lexical_alignments


class Token(object):
    """
    Simple data container class representing a token and its linguistic
    annotations.
    """
    __slots__ = ['id', 'text', 'pos', 'lemma', 'head', 'dependents',
                 'dependency_relation', 'dependency_index', 'word_index']

    def __init__(self, num, text, pos=None, lemma=None):
        self.id = num  # sequential id in the sentence
        self.text = text
        self.pos = pos
        self.lemma = lemma
        self.dependents = []
        self.dependency_relation = None
        self.dependency_index = None
        self.word_index = None

        # Token.head points to another token, not an index
        self.head = None

    def __repr__(self):
        repr_str = '<Token %s, Dep rel=%s>' % (self.text,
                                               self.dependency_relation)
        return _compat_repr(repr_str)

    def __str__(self):
        return _compat_repr(self.text)

    def get_dependent(self, relation, error_if_many=False):
        """
        Return the modifier (syntactic dependents) that has the specified
        dependency relation. If `error_if_many` is true and there is more
        than one have the same relation, it raises a ValueError. If there
        are no dependents with this relation, return None.

        :param relation: the name of the dependency relation
        :param error_if_many: whether to raise an exception if there is
            more than one value
        :return: Token
        """
        deps = [dep for dep in self.dependents
                if dep.dependency_relation == relation]

        if len(deps) == 0:
            return None
        elif len(deps) == 1 or not error_if_many:
            return deps[0]
        else:
            msg = 'More than one dependent with relation {} in token {}'.\
                format(relation, self)
            raise ValueError(msg)

    def get_dependents(self, relation):
        """
        Return modifiers (syntactic dependents) that have the specified dependency
        relation.

        :param relation: the name of the dependency relation
        """
        deps = [dep for dep in self.dependents
                if dep.dependency_relation == relation]

        return deps


class ConllPos(object):
    """
    Dummy class to store field positions in a CoNLL-like file
    for dependency parsing. NB: The positions are different from
    those used in SRL!
    """
    id = 0
    word = 1
    lemma = 2
    pos = 3
    pos2 = 4
    morph = 5
    dep_head = 6  # dependency head
    dep_rel = 7  # dependency relation


class Dependency(object):
    """
    Class to store data about a dependency relation and provide
    methods for comparison
    """
    __slots__ = 'label', 'head', 'dependent'

    def __init__(self, label, head, dependent):
        self.label = label
        self.head = head
        self.dependent = dependent

    def get_data(self):
        head = self.head.lemma if self.head else None
        return self.label, head, self.dependent.lemma

    def __repr__(self):
        s = '{}({}, {})'.format(*self.get_data())
        return _compat_repr(s)

    def __hash__(self):
        return hash(self.get_data())

    def __eq__(self, other):
        """
        Check if the lemmas of head and modifier are the same across
        two Dependency objects.
        """
        if not isinstance(other, Dependency):
            return False
        return self.get_data() == other.get_data()

    def is_equivalent(self, other):
        """
        Return True if this dependency and the other have the same label and
        their three components either have the same lemma or are synonyms.

        :param other: another dependency instance
        :return: boolean
        """
        if self.label != other.label:
            return False

        lemma1 = self.head.lemma if self.head else None
        lemma2 = other.head.lemma if other.head else None
        if lemma1 != lemma2 and not own.are_synonyms(lemma1, lemma2):
            return False

        lemma1 = self.dependent.lemma
        lemma2 = other.dependent.lemma
        if lemma1 != lemma2 and not own.are_synonyms(lemma1, lemma2):
            return False

        return True


class Sentence(object):
    """
    Class to store a sentence with linguistic annotations.
    """
    __slots__ = 'tokens', 'root', 'lower_content_tokens', 'dependencies'

    def __init__(self, parser_output):
        """
        Initialize a sentence from the output of one of the supported parsers. 
        It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.

        :param parser_output: if None, an empty Sentence object is created.
        :param lower: whether to convert tokens to lower case
        """
        self.tokens = []
        self.dependencies = []
        self.root = None
        self.lower_content_tokens = []
        self._read_conll_output(parser_output)
        self._extract_dependency_tuples()

    def __str__(self):
        return ' '.join(str(t) for t in self.tokens)
    
    def __repr__(self):
        repr_str = str(self)
        return _compat_repr(repr_str)

    def _extract_dependency_tuples(self):
        '''
        Extract dependency tuples in the format relation(token1, token2)
        from the sentence tokens.

        These tuples are stored in the sentence object as namedtuples
        (relation, head, modifier). They are stored in a set, so duplicates will
        be lost.
        '''
        self.dependencies = []
        # TODO: use collapsed dependencies (collapse preposition and/or conjunctions)
        for token in self.tokens:
            # ignore punctuation dependencies
            relation = token.dependency_relation
            if relation == 'p':
                continue

            head = token.head
            dep = Dependency(relation, head, token)
            self.dependencies.append(dep)

    def structure_representation(self):
        """
        Return a CoNLL representation of the sentence's syntactic structure.
        """
        lines = []
        for token in self.tokens:
            head = token.head.id if token.head is not None else 0
            lemma = token.lemma if token.lemma is not None else '_'
            line = '{token.id}\t\t{token.text}\t\t{lemma}\t\t{token.pos}\t\t' \
                   '{head}\t\t{token.dependency_relation}'
            line = line.format(token=token, lemma=lemma, head=head)
            lines.append(line)

        return '\n'.join(lines)

    def find_lower_content_tokens(self, stopwords):
        '''
        Store the lower case content tokens (i.e., not in stopwords) for faster
        processing.

        :param stopwords: set
        '''
        self.lower_content_tokens = [token.text.lower()
                                     for token in self.tokens
                                     if token.lemma not in stopwords]

    def _read_conll_output(self, conll_output):
        """
        Internal function to load data in conll dependency parse syntax.
        """
        lines = conll_output.splitlines()
        sentence_heads = []
        
        for line in lines:
            fields = line.split()
            if len(fields) == 0:
                break

            id_ = int(fields[ConllPos.id])
            word = fields[ConllPos.word]
            pos = fields[ConllPos.pos]
            if pos == '_':
                # some systems output the POS tag in the second column
                pos = fields[ConllPos.pos2]

            lemma = fields[ConllPos.lemma]
            if lemma == '_':
                lemma = lemmatization.get_lemma(word, pos)

            head = int(fields[ConllPos.dep_head])
            dep_rel = fields[ConllPos.dep_rel]
            
            # -1 because tokens are numbered from 1
            head -= 1
            
            token = Token(id_, word, pos, lemma)
            token.dependency_relation = dep_rel

            self.tokens.append(token)
            sentence_heads.append(head)
            
        # now, set the head of each token
        for modifier_idx, head_idx in enumerate(sentence_heads):
            # skip root because its head is -1
            if head_idx < 0:
                self.root = self.tokens[modifier_idx]
                continue
            
            head = self.tokens[head_idx]
            modifier = self.tokens[modifier_idx]
            modifier.head = head
            head.dependents.append(modifier)
