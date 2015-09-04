# -*- coding: utf-8 -*-

'''
This module contains data structures used by the related scripts.
'''

import re

class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, t, h, entailment, **attribs):
        '''
        :param entailment: boolean
        :param attribs: extra attributes to be written to the XML
        '''
        self.t = t
        self.h = h
        self.t_attribs = {}
        self.h_attribs = {}
        self.attribs = attribs
        self.attribs['entailment'] = entailment
        self.annotated_h = None
        self.annotated_t = None

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
        return '<Token %s (Lemma: %s, POS: %s)>' % (self.text, self.lemma, self.pos)

class Sentence(object):
    '''
    Class to store a sentence with linguistic annotations.
    '''
    def __init__(self, corenlp_output):
        '''
        Initialize a sentence from the output of the stanford corenlp
        processor. It checks for the tokens themselves, pos tags, lemmas
        and dependency annotations.
        '''
        # ignore the first two lines (they contain the number of tokens and the sentence)
        lines = corenlp_output.splitlines()[2:]
        
        self.tokens = []
        
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
        