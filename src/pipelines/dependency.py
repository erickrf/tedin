# -*- coding: utf-8 -*-

'''
RTE configuration based on simple features extracted
from dependency trees.
'''

import numpy as np
import sklearn.linear_model as linear

from base_configuration import BaseConfiguration
import feature_extraction as fe
import utils


class DependencyPipeline(BaseConfiguration):

    def __init__(self, parser, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        """
        Initialize an OverlapPipeline object with the given data.

        :param parser: which parser to call. Currently supports
            'corenlp', 'malt' and 'palavras'
        :param stopwords: list of stopwords or path to a file
        """
        self.parser = parser
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    def extract_features(self, pairs):
        utils.preprocess_dependency(pairs, self.parser)

        extractors = [fe.dependency_overlap, fe.negation_check,
                      fe.quantity_agreement]

        # some feature extractors return tuples, others return ints
        # convert each one to numpy and then join them
        feature_arrays = [[np.array(f(pair)) for f in extractors]
                          for pair in pairs]

        return feature_arrays



