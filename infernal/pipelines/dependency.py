# -*- coding: utf-8 -*-

'''
RTE configuration based on simple features extracted
from dependency trees.
'''

from __future__ import absolute_import

import sklearn.linear_model as linear

from pipelines.base_configuration import BaseConfiguration
import feature_extraction as fe


class DependencyPipeline(BaseConfiguration):

    def __init__(self, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        """
        Initialize a DependencyPipeline object with the given data.

        :param stopwords: list of stopwords or path to a file
        """
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    @property
    def extractors(self):
        extractors = [lambda p: fe.word_overlap_proportion(p, self.stopwords),
                      lambda p: fe.matching_verb_arguments(p, False),
                      lambda p: fe.bleu(p, False),
                      fe.dependency_overlap,
                      fe.negation_check,
                      fe.quantity_agreement,
                      fe.has_nominalization]
        return extractors
