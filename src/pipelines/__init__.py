from __future__ import absolute_import
from pipelines.overlap import OverlapPipeline
from pipelines.dependency import DependencyPipeline
from pipelines.base_configuration import BaseConfiguration


def get_pipeline(name):
    """
    Return the appropriate pipeline class according to the name.
    """
    if name == 'dependency':
        return DependencyPipeline
    elif name == 'overlap':
        return OverlapPipeline
    else:
        raise ValueError('Unkown pipeline: %s' % name)
