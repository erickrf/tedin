# -*- coding: utf-8 -*-

from __future__ import absolute_import

'''
RTE configuration based on simple features extracted
from word embeddings.
'''

import sklearn.linear_model as linear
import os
import abc
import numpy as np

import utils
from pipelines.base_configuration import BaseConfiguration
import feature_extraction as fe


class BaseEmbedding(BaseConfiguration):
    '''
    Basic pipeline configuration using word embeddings.
    '''
    __metaclass__ = abc.ABCMeta
    oov_vector_filename = 'oov.npy'

    def __init__(self, wd, embeddings, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        """
        Initialize an BaseEmbedding object with the given data.

        :param wd: path to vocabulary file
        :param embeddings: path to numpy embeddings file
        :param stopwords: list of stopwords or path to a file
        :param embeddings: 2-d numpy array
        """
        self.embeddings = utils.EmbeddingDictionary(wd, embeddings)
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    def save(self, dirname):
        super(BaseEmbedding, self).save(dirname)
        path = os.path.join(dirname, self.oov_vector_filename)
        oov_vector = self.embeddings.get_oov_vector()
        np.save(path, oov_vector)

    def load(self, dirname):
        super(BaseEmbedding, self).load(dirname)
        path = os.path.join(dirname, self.oov_vector_filename)
        oov_vector = np.load(path)
        self.embeddings.set_oov_vector(oov_vector)


class EmbeddingOverlap(BaseEmbedding):
    '''
    Simple pipeline that uses as features the embedding overlap:
    the cosine of two word embeddings mean how much they overlap.
    '''
    @property
    def extractors(self):
        return [lambda p: fe.soft_word_overlap(p, self.embeddings)]

