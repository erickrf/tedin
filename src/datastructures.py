# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
This module contains data structures used by the related scripts.
'''

import re
import six
from enum import Enum
import numpy as np

import lemmatization


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
    Class for storing data in the format used by TED models.
    """

    def __init__(self, nodes1, nodes2, sizes1, sizes2, key_roots1,
                 key_roots2, num_key_roots1, num_key_roots2, lmd1, lmd2):
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.sizes1 = sizes1
        self.sizes2 = sizes2
        self.key_roots1 = key_roots1
        self.key_roots2 = key_roots2
        self.num_key_roots1 = num_key_roots1
        self.num_key_roots2 = num_key_roots2
        self.lmd1 = lmd1
        self.lmd2 = lmd2
        self.num_items = len(nodes1)

        # variables in the order they are given in the constructor
        self._ordered_variables = [self.nodes1, self.nodes2, self.sizes1,
                                   self.sizes2, self.key_roots1,
                                   self.key_roots2, self.num_key_roots1,
                                   self.num_key_roots2, self.lmd1, self.lmd2]

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        arrays = [a[item] for a in self._ordered_variables]
        return Dataset(*arrays)


# define an enum with possible entailment values
class Entailment(Enum):
    none = 1
    entailment = 2
    paraphrase = 3
    contradiction = 4


class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
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
    
    def __unicode__(self):
        '''
        Return both sentences
        '''
        return 'T: {}\nH: {}'.format(self.t, self.h)

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
        p.lexical_alignments = [(t2, t1)
                                for (t1, t2) in self.lexical_alignments]
        p.annotated_t = self.annotated_h
        p.annotated_h = self.annotated_t
        return p


class Token(object):
    '''
    Simple data container class representing a token and its linguistic
    annotations.
    '''
    def __init__(self, num, text, pos, lemma=None):
        self.id = num  # sequential id in the sentence
        self.text = text
        self.pos = pos
        self.lemma = lemma
        self.dependents = []
        self.dependency_relation = None
        # Token.head points to another token, not an index
        self.head = None 
    
    def __repr__(self):
        repr_str = '<Token %s, POS=%s>' % (self.text, self.pos)
        return _compat_repr(repr_str)

    def __unicode__(self):
        return self.text

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
        '''
        Return modifiers (syntactic dependents) that have the specified dependency
        relation.

        :param relation: the name of the dependency relation
        '''
        deps = [dep for dep in self.dependents 
                if dep.dependency_relation == relation]

        return deps

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


class Dependency(object):
    """
    Class to store data about a dependency relation and provide
    methods for comparison
    """
    def __init__(self, label, head, modifier):
        self.label = label
        self.head = head
        self.modifier = modifier

    def get_data(self):
        head = self.head.lemma if self.head else None
        return self.label, head, self.modifier.lemma

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


class Sentence(object):
    '''
    Class to store a sentence with linguistic annotations.
    '''
    def __init__(self, text, parser_output=None, output_format='corenlp'):
        '''
        Initialize a sentence from the output of one of the supported parsers. 
        It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.

        :param text: The non-tokenized text of the sentence
        :param parser_output: if None, an empty Sentence object is created.
        :param output_format: 'corenlp', 'palavras' or 'conll'
        '''
        self.tokens = []
        self.text = text
        if parser_output is None:
            return
        
        output_format = output_format.lower()

        if output_format == 'corenlp':
            self._read_corenlp_output(parser_output)
        elif output_format == 'palavras':
            self._read_palavras_output(parser_output)
        elif output_format == 'conll':
            self._read_conll_output(parser_output)
        else:
            raise ValueError('Unknown format: %s' % output_format)
        
        self.extract_dependency_tuples()
        self.lower_content_tokens = None

    def find_lower_content_tokens(self, stopwords):
        '''
        Store the lower case content tokens (i.e., not in stopwords) for faster
        processing.

        :param stopwords: set
        '''
        self.lower_content_tokens = [token.text.lower()
                                     for token in self.tokens
                                     if token.lemma not in stopwords]

    def __unicode__(self):
        return ' '.join(unicode(t) for t in self.tokens)

    def __str__(self):
        return ' '.join(str(t) for t in self.tokens)
    
    def __repr__(self):
        repr_str = str(self)
        return _compat_repr(repr_str)
    
    def extract_dependency_tuples(self):
        '''
        Extract dependency tuples in the format relation(token1, token2)
        from the sentence tokens.
        
        These tuples are stored in the sentence object as namedtuples
        (relation, head, modifier). They are stored in a set, so duplicates will be lost.
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
        '''
        Return a CoNLL-like representation of the sentence's syntactic structure.
        '''
        lines = []
        for token in self.tokens:
            head = token.head.id if token.head is not None else 0
            lemma = token.lemma if token.lemma is not None else '_'
            line = '{token.id}\t\t{token.text}\t\t{lemma}\t\t{token.pos}\t\t' \
                   '{head}\t\t{token.dependency_relation}'
            line = line.format(token=token, lemma=lemma, head=head)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def find_lemmas(self):
        '''
        Find the lemmas for all tokens in the sentence.
        '''
        pos_to_check = ['NOUN', 'VERB', 'ADJ']
        for token in self.tokens:
            if token.pos in pos_to_check:
                token.lemma = lemmatization.get_lemma(token.text, token.pos)
            else:
                token.lemma = token.text
    
    def _read_palavras_output(self, palavras_output):
        '''
        Internal function to load data from the output of the Palavras parser
        for Portuguese.
        '''
        palavras_output = palavras_output.decode('utf-8')
        lines = palavras_output.splitlines()
        dependencies = {}
        
        for line  in lines:
            if line == '</s>':
                # palavras output for a sentence ends with this tag
                break
            
            parts = line.split()
            
            # punctuation usually only has the token preceded by $ and the
            # dep rel
            if len(parts) == 2:
                token = self._read_palavras_punctuation(parts, dependencies)
            else: 
                token = self._read_palavras_token(parts, dependencies)
            
            self.tokens.append(token)
            
        # add dependency heads
        for modifier_index in dependencies:
            head_index = dependencies[modifier_index]
            if head_index == -1:
                # this is the root
                self.head_index = head_index
                continue
            
            head = self.tokens[head_index]
            modifier = self.tokens[modifier_index]
            
            head.dependents.append(modifier)
            modifier.head = head
    
    def _read_palavras_token(self, parts, dependencies):
        '''
        :param parts: one line of the palavras output, split at whitespaces
        :param dependencies: dictionary of (modifer -> head) to be updated
        '''
        text = parts[0]
        # lemma is surrounded by []
        lemma = parts[1][1:-1]
        
        # the POS tag is the first part after info between angle brackets
        index = 2
        while parts[index][0] == '<':
            index += 1
        pos = parts[index]
        token = Token(None, text, pos, lemma)
        
        dep_rel = parts[-2]
        token.dependency_relation = dep_rel
        
        # store dependency information. we add it to the token objects after
        # all have been created
        head, modifier = self._read_palavras_dependency(parts[-1])
        dependencies[modifier] = head
        
        return token   
    
    def _read_palavras_punctuation(self, parts, dependencies):
        '''
        :param parts: one line of the palavras output, split at whitespaces
        :param dependencies: dictionary of (modifer -> head) to be updated
        '''
        text = parts[0]
        if text[0] == '$':
            # punctuation signs are preceded by $, but not numbers
            text = text[1:]
        
        pos = 'punct'
        dep_rel = 'punct'
        head, modifier = self._read_palavras_dependency(parts[1])
        dependencies[modifier] = head
        token = Token(None, text, pos, text)
        token.dependency_relation = dep_rel
        
        return token
        
    def _read_palavras_dependency(self, dep_string):
        '''
        Read a dependency link in the palavras notation, in the form #1->3
        It returns a tuple (head, modifier) as ints, both subtracted by 1 
        (because indices start from 1 in the palavras output).
        '''
        dep_arrow = dep_string[1:]
        modifier, head = dep_arrow.split('->')
        
        # indices start from 1 in the parser output
        modifier = int(modifier) - 1
        head = int(head) - 1
        
        return (head, modifier)
    
    def _read_conll_output(self, conll_output):
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
    
    def _read_corenlp_output(self, corenlp_output):
        '''
        Internal function to load data from the output of the Stanford
        corenlp processor.
        '''
        # ignore the first two lines (they contain the number of tokens
        # and the sentence)
        lines = corenlp_output.splitlines()[2:]
        
        token_regex = r'Text=(.+) CharacterOffsetBegin.+ PartOfSpeech=(.+)\]'
        dependency_regex = r'(\w+)\(.+-(\d+), .+-(\d+)\)'
        token_num = 1
        for line in lines:
            if line.strip() == '':
                    continue
                
            elif line.startswith('['):
                match = re.search(token_regex, line)
                text, pos = match.groups()
                lemma = lemmatization.get_lemma(text, pos)
                token = Token(token_num, text, pos, lemma)
                self.tokens.append(token)
                token_num += 1
            
            else:
                # dependency information
                match = re.search(dependency_regex, line)
                relation, head_num, modifier_num = match.groups()
                
                # head and modifier have the index of the token in the sentence,
                # starting from 1
                head_num = int(head_num) - 1
                modifier_num = int(modifier_num) - 1
                
                modifier = self.tokens[modifier_num]
                modifier.dependency_relation = relation
                
                if head_num >= 0:
                    # do not to this for the root (its head index is -1)
                    head = self.tokens[head_num]
                    modifier.head = head
                    head.dependents.append(modifier)
                else:
                    self.root = modifier
