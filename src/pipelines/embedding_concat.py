# -*- coding: utf-8 -*-

'''
RTE configuration based on concatenating embedding representations
of two sentences.
'''

from simple_embedding import BaseEmbedding
import feature_extraction as fe


class EmbeddingConcat(BaseEmbedding):
    '''
    Pipeline class that concatenate embedding representations for
    two sentences as features.
    '''
    @property
    def extractors(self):
        return [lambda p: fe.sentence_average_embeddings(p, self.embeddings)]
