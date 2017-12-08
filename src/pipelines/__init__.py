from __future__ import absolute_import
from pipelines.overlap import OverlapPipeline
from pipelines.dependency import DependencyPipeline
from pipelines.base_configuration import BaseConfiguration
from pipelines.simple_embedding import EmbeddingOverlap, BaseEmbedding
from pipelines.embedding_concat import EmbeddingConcat
from pipelines.similarities import SimilarityPipeline

pipeline_dict = {'dependency': DependencyPipeline,
                 'overlap': OverlapPipeline,
                 'embedding': EmbeddingConcat,
                 'similarity': SimilarityPipeline}

pipeline_names = list(pipeline_dict.keys())


def get_pipeline(name):
    """
    Return the appropriate pipeline class according to the name.
    """
    class_ = pipeline_dict.get(name, None)
    if class_ is None:
        raise ValueError('Unkown pipeline: %s' % name)

    return class_
