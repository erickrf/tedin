# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

"""
Classes and functions for dealing with data from the paraphrase database (PPDB)
"""


class TransformationDict(object):
    """
    Class storing a dictionary for phrasal, lexical and/or syntactic
    transformations.
    """
    def __init__(self):
        # each key in d is a token, and each value is a tuple (set, dict)
        # the set is the set of RHS in the rule, and the dict has the same
        # structure as d
        self.d = dict()

    def add(self, lhs, rhs):
        """
        Add a transformation rule.

        :param lhs: left-hand side, tuple/list of strings
        :param rhs: left-hand side, tuple/list of strings
        """
        d = self.d
        for token in lhs:
            if token in d:
                rule_set, d = d[token]
            else:
                rule_set = {}
                new_d = dict()
                d[token] = (rule_set, new_d)
                d = new_d

        rule_set.add(rhs)
