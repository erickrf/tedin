# -*- coding: utf-8 -*-

'''
RTE configuration based on simple features extracted
from dependency trees.
'''

import feature_extraction as fe

extract_features = fe.pipeline_dependency
parser = 'corenlp'
