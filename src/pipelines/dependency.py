# -*- coding: utf-8 -*-

'''
RTE configuration based on simple features extracted
from dependency trees.
'''

from __future__ import absolute_import

import numpy as np
import sklearn.linear_model as linear

from pipelines.base_configuration import BaseConfiguration
import feature_extraction as fe


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

    def extract_features(self, pairs, preprocessed=True):
        extractors = [lambda p: fe.word_overlap_proportion(p, self.stopwords),
                      lambda p: fe.matching_verb_arguments(p, False),
                      lambda p: fe.bleu(p, False),
                      fe.dependency_overlap,
                      fe.negation_check,
                      fe.quantity_agreement,
                      fe.has_nominalization]
        all_features = []

        # some feature extractors return tuples, others return ints
        # convert each one to numpy and then ensure all are 2-dim
        new_shape = (len(pairs), -1)
        for func in extractors:
            feature_values = np.array([func(pair) for pair in pairs])
            feature_values = feature_values.reshape(new_shape)
            all_features.append(feature_values)

        features = np.hstack(all_features)
        return features



