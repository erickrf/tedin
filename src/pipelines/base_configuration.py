# -*- coding: utf-8 -*-

'''
Base class with pipeline configuration for training models.
'''

import abc
import os
import cPickle


class BaseConfiguration:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.classifier = None
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def extract_features(self, pairs):
        pass

    @abc.abstractmethod
    def train_classifier(self, pairs):
        pass

    @abc.abstractmethod
    def classify(self, pairs):
        pass

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
