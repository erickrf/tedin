# -*- coding: utf-8 -*-

'''
Configuration for the RTE system based only on word overlap.
'''

import feature_extraction as fe

extract_features = fe.pipeline_minimal
stopwords_path = 'data/stopwords.txt'

# ==================================
# Machine learning algorithms config
# ==================================

import sklearn.linear_model as linear
classifier_class = linear.LogisticRegression
regressor_class = linear.LinearRegression

classifier_parameters = {'class_weight': 'auto'}
regressor_parameters = {}