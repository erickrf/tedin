# -*- coding: utf-8 -*-

"""
RTE baseline based on similarity metrics between two sentences:

- TF-IDF similarity
- Average of per-word max embedding cosine similarity
- Proportion of words
- Proportion of size
"""

from pipelines import BaseConfiguration
import feature_extraction as fe
import utils

import sklearn.linear_model as linear


class SimilarityPipeline(BaseConfiguration):

    def __init__(self, wd, embeddings, stopwords=None,
                 classifier_class=linear.LogisticRegression,
                 classifier_parameters={'class_weight': 'balanced'}):
        self.embeddings = utils.EmbeddingDictionary(wd, embeddings)
        self._load_stopwords(stopwords)
        self.classifier = classifier_class(**classifier_parameters)

    @property
    def extractors(self):
        extractors = [lambda p: fe.word_overlap_proportion(p, self.stopwords),
                      lambda p: fe.soft_word_overlap(p, self.embeddings),
                      fe.length_proportion]
        return extractors
