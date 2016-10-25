# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Script to evaluate a trained sklearn model on a RTE dataset.
'''

import argparse
import os
import cPickle
import sklearn
from scipy.stats import pearsonr, spearmanr

import pipelines
import utils


def eval_regressor(data_dir, x, y):
    '''
    Evaluate regressor Pearson and R^2 scores
    '''
    filename = os.path.join(data_dir, 'regressor.dat')
    with open(filename, 'rb') as f:
        regressor = cPickle.load(f)
    
    predicted = regressor.predict(x)
    pearson = pearsonr(y, predicted)[0] * 100
    spearman = spearmanr(predicted, y)[0] * 100
    diff = predicted - y
    mse = (diff ** 2).mean()
    
    print('Similarity evaluation')
    print('Pearson   Spearman   Mean Squared Error')
    print('-------   --------   ------------------')
    print('{:7.2f}   {:8.2f}   {:18.2f}'.format(pearson, spearman, mse))
    

def eval_rte(pairs, pipeline_name, model, use_stopwords):
    """
    Evaluate the RTE pipeline.
    :param pairs: list of Pair objects
    :param pipeline_name: name of the pipeline
    :param model: path to saved model
    :param use_stopwords: whether to use a stopword list
    """
    inverted_pairs = [p.inverted_pair() for p in pairs]

    pipeline_class = pipelines.get_pipeline(pipeline_name)
    stopwords = utils.load_stopwords() if use_stopwords else None
    pipeline = pipeline_class(stopwords=stopwords)

    assert isinstance(pipeline, pipelines.BaseConfiguration)
    pipeline.load(model)
    classifier = pipeline.classifier

    # we classify the pairs in both ways with binary classes (entailment or not)
    # if both trigger, it is a paraphrase.
    x = pipeline.extract_features(pairs)
    x_inv = pipeline.extract_features(inverted_pairs)
    y = utils.extract_classes(pairs)

    predictions_original = classifier.predict(x)
    predictions_inverted = classifier.predict(x_inv)
    final_predictions = utils.combine_paraphrase_predictions(predictions_original,
                                                             predictions_inverted)
    macro_f1 = sklearn.metrics.f1_score(y, final_predictions,
                                        average='macro') * 100
    accuracy = sklearn.metrics.accuracy_score(y, final_predictions)

    print('RTE evaluation')
    print('Accuracy\tMacro F1')
    print('--------\t--------')
    print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Directory with trained model')
    parser.add_argument('test_file', help='File with preprocessed test data')
    parser.add_argument('pipeline', help='Which pipeline to use',
                        choices=['dependency', 'overlap'])
    parser.add_argument('-s', help='Use stopwords', action='store_true',
                        dest='use_stopwords')
    args = parser.parse_args()

    with open(args.test_file, 'rb') as f:
        pairs = cPickle.load(f)

    eval_rte(pairs, args.pipeline, args.input, args.use_stopwords)

