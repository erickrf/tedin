# -*- coding: utf-8 -*-

'''
Configuration for the RTE system based only on word overlap.
'''

import sklearn.linear_model as linear

from base_configuration import BaseConfiguration
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

    def _load_stopwords(self, stopwords):
        if stopwords is None:
            self.stopwords = None
        elif isinstance(stopwords, list):
            self.stopwords = stopwords
        else:
            with open(stopwords, 'rb') as f:
                text = unicode(f.read(), 'utf-8')

            self.stopwords = set(text.splitlines())

    def extract_features(self, pairs):
        utils.tokenize_pairs(pairs, lower=True)
        return fe.word_overlap_proportion(pairs, self.stopwords)

    def train_classifier(self, pairs):
        features = self.extract_features(pairs)
        labels = utils.extract_classes(pairs)
        self.classifier.fit(features, labels)

    def classify(self, pairs):
        features = self.extract_features(pairs)
        return self.classifier.predict(features)

