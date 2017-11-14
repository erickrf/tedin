# -*- coding: utf-8 -*-

'''
Base class with pipeline configuration for training models.
'''

from __future__ import absolute_import

import abc
import os
import cPickle
import numpy as np

import utils


class BaseConfiguration:
    __metaclass__ = abc.ABCMeta

    classifier_filename = 'classifier.pickle'
    stopwords_filename = 'stopwords.pickle'

    def __init__(self):
        self.classifier = None
        raise NotImplementedError('This is an abstract class')

    @abc.abstractproperty
    def extractors(self):
        '''
        This is a list of functions that can be called with the pair
        as the only argument. Each one return one or more values.
        '''
        return []

    def extract_features(self, pairs, preprocessed=True):
        '''
        Extracts features from the pairs and returns a 2-d
        numpy array with their values
        '''
        all_features = []

        # some feature extractors return tuples, others return ints
        # convert each one to numpy and then ensure all are 2-dim
        new_shape = (len(pairs), -1)
        for func in self.extractors:
            feature_values = np.array([func(pair) for pair in pairs])
            feature_values = feature_values.reshape(new_shape)
            all_features.append(feature_values)

        features = np.hstack(all_features)
        return features

    def _load_stopwords(self, stopwords):
        if stopwords is None:
            self.stopwords = set()

        elif isinstance(stopwords, list):
            self.stopwords = set(stopwords)
        else:
            if isinstance(stopwords, basestring):
                with open(stopwords, 'rb') as f:
                    text = unicode(f.read(), 'utf-8')
                self.stopwords = set(text.splitlines())
            else:
                self.stopwords = stopwords

        self.stopwords.update(['.', ',', ';', ':', '(', ')', "'", '"',
                               '!', '-', '--'])

    def train_classifier(self, pairs, preprocessed=True):
        """
        Extract features and train a classifier.
        :param pairs: list of `Pair` objects
        :param preprocessed: whether pairs have already been preprocessed
        :return:
        """
        features = self.extract_features(pairs, preprocessed)
        labels = utils.extract_classes(pairs)
        self.classifier.fit(features, labels)

    def classify(self, pairs):
        features = self.extract_features(pairs)
        return self.classifier.predict(features)

    def save(self, dirname):
        '''
        Save the classifier attribute to a file.
        Other classes should extend this method when necessary.

        :param dirname: directory where the classifier will be saved.
        '''
        path = os.path.join(dirname, self.classifier_filename)
        with open(path, 'wb') as f:
            cPickle.dump(self.classifier, f, -1)

        path = os.path.join(dirname, self.stopwords_filename)
        with open(path, 'wb') as f:
            cPickle.dump(self.stopwords, f, -1)

    def load(self, dirname):
        '''
        Load the classifier attribute from a file.

        Other classes should extend this method when necessary.

        :param dirname: directory where the classifier is saved.
        '''
        path = os.path.join(dirname, self.classifier_filename)
        with open(path, 'rb') as f:
            self.classifier = cPickle.load(f)

        path = os.path.join(dirname, self.stopwords_filename)
        with open(path, 'rb') as f:
            self.stopwords = cPickle.load(f)
