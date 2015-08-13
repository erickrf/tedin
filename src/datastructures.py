# -*- coding: utf-8 -*-

'''
This module contains data structures used by the related scripts.
'''


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
        self.entailment = entailment
        self.attribs = attribs
