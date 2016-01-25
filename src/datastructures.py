# -*- coding: utf-8 -*-

'''
This module contains data structures used by the related scripts.
'''

from __future__ import unicode_literals
import re
import six
from enum import Enum
from collections import namedtuple

import resources

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
    def __init__(self, t, h, id_, entailment, similarity=None):
        '''
        :param entailment: boolean
        :param attribs: extra attributes to be written to the XML
        '''
        self.t = t
        self.h = h
        self.id = id_
        self.lexical_alignments = None
        self.entailment = entailment
        self.annotated_h = None
        self.annotated_t = None
        
        if similarity is not None:
            self.similarity = similarity
    
    def __unicode__(self):
        '''
        Return both sentences
        '''
        return u'T: {}\nH: {}'.format(self.t, self.h)

class Token(object):
    '''
    Simple data container class representing a token and its linguistic
    annotations.
    '''
    def __init__(self, text, pos, lemma=None):
        self.text = text
        self.pos = pos
        self.lemma = lemma
        self.dependents = []
        self.dependency_relation = None
        # Token.head points to another token, not an index
        self.head = None 
    
    def __repr__(self):
        repr_str = u'<Token %s>' % self.text
        return _compat_repr(repr_str)
    
    def __unicode__(self):
        return u'<Token %s>' % self.text

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

Dependency = namedtuple('Dependency', ['relation', 'head', 'modifier'])

class Sentence(object):
    '''
    Class to store a sentence with linguistic annotations.
    '''
    def __init__(self, parser_output, output_format='corenlp'):
        '''
        Initialize a sentence from the output of one of the supported parsers. 
        It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.
        
        :param parser_output: if None, an empty Sentence object is created.
        :param output_format: 'corenlp', 'palavras' or 'conll'
        '''
        if parser_output is None:
            return
        
        output_format = output_format.lower()
        self.tokens = []
        
        if output_format == 'corenlp':
            self._read_corenlp_output(parser_output)
        elif output_format == 'palavras':
            self._read_palavras_output(parser_output)
        elif output_format == 'conll':
            self._read_conll_output(parser_output)
        else:
            raise ValueError('Unknown format: %s' % output_format)
        
        self.extract_dependency_tuples()
    
    def __unicode__(self):
        return ' '.join(unicode(t) for t in self.tokens)
    
    def __repr__(self):
        repr_str = unicode(self)
        return _compat_repr(repr_str)
    
    def extract_dependency_tuples(self):
        '''
        Extract dependency tuples in the format relation(token1, token2)
        from the sentence tokens.
        
        These tuples are stored in the sentence object as namedtuples
        (relation, head, modifier). They are stored in a set, so duplicates will be lost.
        '''
        self.dependencies = set()
        # TODO: use collapsed dependencies (collapse preposition and/or conjunctions)
        for token in self.tokens:
            # ignore punctuation dependencies
            relation = token.dependency_relation
            if relation == 'p':
                continue
            
            head = 'ROOT' if token.head is None else token.head.text.lower()
            dep = Dependency(relation, head, token.text)
            self.dependencies.add(dep)
    
    def structure_representation(self):
        '''
        Return a CoNLL-like representation of the sentence's syntactic structure.
        '''
        lines = []
        for token in self.tokens:
            head = token.head.text if token.head is not None else 'root'
            lemma = token.lemma if token.lemma is not None else '_'
            line = '{token.text}\t\t{lemma}\t\t{token.pos}\t\t{head}\t\t{token.dependency_relation}'
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
                token.lemma = resources.get_lemma(token.text, token.pos)
            else:
                token.lemma = token.text
    
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
            
            word = fields[ConllPos.word]
            pos = fields[ConllPos.pos]
            if pos == '_':
                # some systems output the POS tag in the second column
                pos = fields[ConllPos.pos2]
            
            lemma = fields[ConllPos.lemma]
            if lemma == '_':
                lemma = resources.get_lemma(word, pos)
                
            head = int(fields[ConllPos.dep_head])
            dep_rel = fields[ConllPos.dep_rel]
            
            # -1 because tokens are numbered from 1
            head -= 1
            
            token = Token(word, pos, lemma)
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
        Internal function to load data from the output of the Stanford corenlp processor.
        '''
        # ignore the first two lines (they contain the number of tokens and the sentence)
        lines = corenlp_output.splitlines()[2:]
        
        token_regex = r'Text=(.+) CharacterOffsetBegin.+ PartOfSpeech=(.+)\]'
        dependency_regex = r'(\w+)\(.+-(\d+), .+-(\d+)\)'
        for line in lines:
            if line.strip() == '':
                    continue
                
            elif line.startswith('['):
                match = re.search(token_regex, line)
                text, pos = match.groups()
                lemma = resources.get_lemma(text, pos)
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
                else:
                    self.root = modifier
        