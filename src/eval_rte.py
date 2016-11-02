# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Script to evaluate a trained sklearn model on a RTE dataset.
'''

import argparse
import os
import cPickle
import sklearn
import numpy as np
from scipy.stats import pearsonr, spearmanr

import pipelines
import datastructures as ds
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
    

def eval_rte(data_path, pipeline_name, model, binarize):
    """
    Evaluate the RTE pipeline.
    :param data_path: path to saved pairs
    :param pipeline_name: name of the pipeline
    :param model: path to saved model
    :param binarize: convert paraphrases to entailment (2 class problem)
    """
    def print_results(y, pred, name):
        macro_f1 = sklearn.metrics.f1_score(y, pred,
                                            average=None).mean() * 100
        accuracy = sklearn.metrics.accuracy_score(y, pred)

        print('Evaluation on %s' % name)
        print('Accuracy\tMacro F1')
        print('--------\t--------')
        print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))
        print()

    pipeline_class = pipelines.get_pipeline(pipeline_name)
    pipeline = pipeline_class()

    assert isinstance(pipeline, pipelines.BaseConfiguration)
    pipeline.load(model)
    classifier = pipeline.classifier

    pairs = utils.read_pairs(data_path, True, binarize)
    half = len(pairs) / 2
    original_pairs = pairs[:half]
    inverted_pairs = pairs[half:]

    x_original = pipeline.extract_features(original_pairs)
    x_inv = pipeline.extract_features(inverted_pairs)
    y_all = utils.extract_classes(pairs)
    y_original = y_all[:half]
    y_inv = y_all[half:]

    predictions_original = classifier.predict(x_original)
    predictions_inverted = classifier.predict(x_inv)
    predictions_all = np.append(predictions_original, predictions_inverted)
    predictions_combined = utils.combine_paraphrase_predictions(predictions_original,
                                                                predictions_inverted)

    #
    # int_to_name = {1: 'None',
    #                2: 'Entailment',
    #                3: 'Paraphrase'}
    # labels1 = [int_to_name[val] for val in predictions_original]
    # labels2 = [int_to_name[val] for val in predictions_inverted]
    # labels3 = [int_to_name[val] for val in final_predictions]
    # gold_labels = [int_to_name[val] for val in y]
    # text = '\n'.join(','.join(labels)
    #                  for labels in zip(labels1, labels2, labels3, gold_labels))
    # with open('bi.csv', 'wb') as f:
    #     f.write(text)
    print_results(y_original, predictions_original, 'Original dataset')
    print_results(y_inv, predictions_inverted, 'Inverted dataset')
    print_results(y_all, predictions_all, 'Original and inverted overall')
    if not binarize:
        print_results(y_original, predictions_combined, 'Original, combining 2-way')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Directory with trained model')
    parser.add_argument('test_file', help='File with preprocessed test data')
    parser.add_argument('pipeline', help='Which pipeline to use',
                        choices=['dependency', 'overlap'])
    parser.add_argument('-b', action='store_true', dest='binarize',
                        help='Binarize (convert paraphrase'
                             ' to entailment)')
    args = parser.parse_args()

    eval_rte(args.test_file, args.pipeline, args.input, args.binarize)

