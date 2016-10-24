# -*- coding: utf-8 -*-

'''
Base class with pipeline configuration for training models.
'''

from __future__ import absolute_import

import abc
import os
import cPickle

import utils


class BaseConfiguration:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.classifier = None
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def extract_features(self, pairs, preprocessed=True):
        pass

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
        path = os.path.join(dirname, 'classifier.pickle')
        with open(path, 'wb') as f:
            cPickle.dump(self.classifier, f, -1)

    def load(self, dirname):
        '''
        Load the classifier attribute from a file.

        Other classes should extend this method when necessary.

        :param dirname: directory where the classifier is saved.
        '''
        path = os.path.join(dirname, 'classifier.pickle')
        with open(path, 'rb') as f:
            self.classifier = cPickle.load(f)
