# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
This module contains data structures used by the related scripts.
'''

import six
from enum import Enum
import numpy as np
from collections import namedtuple


def _compat_repr(repr_string, encoding='utf-8'):
    '''
    Function to provide compatibility with Python 2 and 3 with the __repr__
    function. In Python 2, return a encoded version. In Python 3, return
    a unicode object. 
    '''
    if six.PY2:
        return repr_string.encode(encoding)
    else:
        return repr_string


class Dataset(object):
    """
    Class for storing data in the format used by TEDIN models.
    """

    def __init__(self, pairs, nodes1, nodes2, labels=None):
        """
        Create a Dataset

        :param pairs: list of Pair objects
        :param nodes1: numpy array (num_pairs, max_len, num_inds) where num_inds
            is the number of indices to represent a node. Usually it is 2:
            word embedding and dependency label embedding.
        :param nodes2: numpy array (num_pairs, max_len, num_inds)
        :param labels: numpy array (num_pairs) or None (this is the target
            class)
        """
        self.pairs = pairs
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.labels = labels
        self.last_batch_index = 0
        self.epoch = 1
        self._post_assignments()

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
            state = np.random.get_state()

    def _post_assignments(self):
        """
        Assign variables to keep track of stuff.
        """
        self.num_items = len(self.nodes1)

        # variables in the order they are given in the constructor
        self._ordered_variables = [self.pairs, self.nodes1, self.nodes2]
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


Pair = namedtuple('Pair', ['annotated_t', 'annotated_h', 'label'])


class Token(object):
    '''
    Simple data container class representing a token and its linguistic
    annotations.
    '''
    __slots__ = 'id', 'index', 'dep_index', 'dependents', 'head'

    def __init__(self, num, index, dep_index):
        self.id = num  # sequential id in the sentence
        self.index = index
        self.dep_index = dep_index
        self.dependents = []

        # Token.head points to another token, not an id
        self.head = None

    def __repr__(self):
        repr_str = '<Token %s, Dep rel=%s>' % (self.index, self.dep_index)
        return _compat_repr(repr_str)

    def __str__(self):
        return _compat_repr('Token %d' % self.index)


class ConllPos(object):
    '''
    Dummy class to store field positions in a CoNLL-like file
    for dependency parsing. NB: The positions are different from
    those used in SRL!
    '''
    id = 0
    word = 1
    lemma = 2
    pos = 3
    pos2 = 4
    morph = 5
    dep_head = 6 # dependency head
    dep_rel = 7 # dependency relation


class Sentence(object):
    '''
    Class to store a sentence with linguistic annotations.
    '''
    __slots__ = 'tokens', 'root'

    def __init__(self, parser_output, word_dict, dep_dict, lower=False):
        '''
        Initialize a sentence from the output of one of the supported parsers. 
        It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.

        :param parser_output: if None, an empty Sentence object is created.
        :param word_dict: dictionary mapping words to integers
        :param dep_dict: dictionary mapping dependency relations to integers
        :param lower: whether to convert tokens to lower case
        '''
        self.tokens = []
        self._read_conll_output(parser_output, word_dict, dep_dict, lower)

    def __str__(self):
        return ' '.join(str(t) for t in self.tokens)
    
    def __repr__(self):
        repr_str = str(self)
        return _compat_repr(repr_str)
    
    def _read_conll_output(self, conll_output, word_dict, dep_dict, lower):
        '''
        Internal function to load data in conll dependency parse syntax.
        '''
        lines = conll_output.splitlines()
        sentence_heads = []
        
        for line in lines:
            fields = line.split()
            if len(fields) == 0:
                break

            id_ = int(fields[ConllPos.id])
            word = fields[ConllPos.word]
            if lower:
                word = word.lower()
            index = word_dict[word]

            head = int(fields[ConllPos.dep_head])
            dep_rel = fields[ConllPos.dep_rel]
            dep_index = dep_dict[dep_rel]
            
            # -1 because tokens are numbered from 1
            head -= 1
            
            token = Token(id_, index, dep_index)
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
