# -*- coding: utf-8 -*-

'''
Configuration for the RTE system based only on word overlap.
'''

from __future__ import absolute_import

import sklearn.linear_model as linear
import numpy as np

from pipelines.base_configuration import BaseConfiguration
import feature_extraction as fe
import utils


class OverlapPipeline(BaseConfiguration):

    def __init__(self, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        """
        Initialize an OverlapPipeline object with the given data.

        :param stopwords: list of stopwords or path to a file
        """
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    def extract_features(self, pairs, preprocessed=True):
        if not preprocessed:
            utils.tokenize_pairs(pairs, lower=True)
        values = [fe.word_overlap_proportion(pair, self.stopwords)
                  for pair in pairs]
        return np.array(values)
