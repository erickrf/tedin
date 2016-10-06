# -*- coding: utf-8 -*-

'''
RTE configuration based on simple features extracted
from dependency trees.
'''

import numpy as np
import traceback
import logging
import sklearn.linear_model as linear

from datastructures import Sentence
from base_configuration import BaseConfiguration
import feature_extraction as fe
import external
import utils


class DependencyPipeline(BaseConfiguration):

    def __init__(self, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        """
        Initialize an OverlapPipeline object with the given data.

        :param stopwords: list of stopwords or path to a file
        """
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    def extract_features(self, pairs):
        extractors = [fe.dependency_overlap, fe.negation_check,
                      fe.quantity_agreement]

        # some feature extractors return tuples, others return ints
        # convert each one to numpy and then join them
        feature_arrays = [[np.array(f(pair)) for f in extractors]
                          for pair in pairs]

        return feature_arrays



