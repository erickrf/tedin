# -*- coding: utf-8 -*-

'''
This module contains data structures used by the related scripts.
'''

from __future__ import unicode_literals
import re
import six
from enum import Enum

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
    def __init__(self, t, h, entailment, similarity=None):
        '''
        :param entailment: boolean
        :param attribs: extra attributes to be written to the XML
        '''
        self.t = t
        self.h = h
        
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

class Token(object):
    '''
    Simple data container class representing a token and its linguistic
    annotations.
    '''
    def __init__(self, text, pos, lemma):
        self.text = text
        self.pos = pos
        self.lemma = lemma
        self.dependents = []
        self.dependency_relation = None
        self.head = None
    
    def __repr__(self):
        repr_str = u'<Token %s (Lemma: %s, POS: %s)>' % (self.text, self.lemma, self.pos)
        return _compat_repr(repr_str)

class Sentence(object):
    '''
    Class to store a sentence with linguistic annotations.
    '''
    def __init__(self, parser_output, parser='corenlp'):
        '''
        Initialize a sentence from the output of one of the supported parsers. 
        It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.
        
        :param parser_output: if None, an empty Sentence object is created.
        :param parser: either 'corenlp' or 'palavras'
        '''
        if parser_output is None:
            return
        
        parser = parser.lower()
        self.tokens = []
        
        if parser == 'corenlp':
            self._read_corenlp_output(parser_output)
        elif parser == 'palavras':
            self._read_palavras_output(parser_output)
        else:
            raise ValueError('Unknown parser: %s' % parser)
    
    def __unicode__(self):
        return ' '.join(self.tokens)
    
    def __repr__(self):
        repr_str = unicode(self)
        return _compat_repr(repr_str)
    
    def _read_palavras_output(self, palavras_output):
        '''
        Internal function to load data from the output of the Palavras parser for Portuguese.
        '''
        palavras_output = unicode(palavras_output, 'utf-8')
        lines = palavras_output.splitlines()
        dependencies = {}
        
        for line  in lines:
            if line == '</s>':
                # palavras output for a sentence ends with this tag
                break
            
            parts = line.split()
            
            # punctuation usually only has the token preceded by $ and the dep rel
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
        token = Token(text, pos, lemma)
        
        dep_rel = parts[-2]
        token.dependency_relation = dep_rel
        
        # store dependency information. we add it to the token objects after all have been created
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
        token = Token(text, pos, text)
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
    
    def _read_corenlp_output(self, corenlp_output):
        '''
        Internal function to load data from the output of the Stanford corenlp processor.
        '''
        # ignore the first two lines (they contain the number of tokens and the sentence)
        lines = corenlp_output.splitlines()[2:]
        
        token_regex = r'Text=(.+) CharacterOffsetBegin.+ PartOfSpeech=(.+) Lemma=(.+)\]'
        dependency_regex = r'(\w+)\(.+-(\d+), .+-(\d+)\)'
        for line in lines:
            if line.strip() == '':
                    continue
                
            elif line.startswith('['):
                match = re.search(token_regex, line)
                text, pos, lemma = match.groups()
                token = Token(text, pos, lemma)
                self.tokens.append(token)
            
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
                    self.root = modifier
        