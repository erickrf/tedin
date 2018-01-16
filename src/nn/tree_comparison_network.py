# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from tree_edit_network import TreeEditDistanceNetwork


class TreeComparisonNetwork(TreeEditDistanceNetwork):
    """
    Class that learns parameters for tree edit distance comparison by training
    on pairs of unrelated and related trees.
    """
    def __init__(self):
        pass
