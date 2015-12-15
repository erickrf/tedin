# -*- coding: utf-8 -*-

'''
Script to evaluate a trained sklearn model on a RTE dataset.
'''

import argparse
import os
import cPickle
import sklearn
from scipy.stats import pearsonr, spearmanr

import utils
import feature_extraction as fe

def eval_classifier(data_dir, x, y):
    '''
    Evaluate classifier accuracy and F1 score.
    '''
    filename = os.path.join(data_dir, 'classifier.dat')
    with open(filename, 'rb') as f:
        classifier = cPickle.load(f)
    
    accuracy = classifier.score(x, y)
    predicted = classifier.predict(x)
    macro_f1 = sklearn.metrics.f1_score(y, predicted, average='macro') * 100
    micro_f1 = sklearn.metrics.f1_score(y, predicted, average='micro') * 100
    
    print 'Classifier accuracy: {:.2%}'.format(accuracy)
    print 'Macro F1: {:.2f}'.format(macro_f1)
    print 'Micro F1: {:.2f}'.format(micro_f1)

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
    
    print 'Pearson correlation: {:.2f}'.format(pearson)
    print 'Spearman correlation (monotonicity): {:.2f}'.format(spearman)
    print 'Mean Squared Error: {:.2f}'.format(mse)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Directory with trained model')
    parser.add_argument('test_file', help='File with test data')
    parser.add_argument('-c', help='Evaluate entailment classifier', action='store_true',
                        dest='classifier')
    parser.add_argument('-r', help='Evaluate similarity regressor', action='store_true',
                        dest='regressor')
    args = parser.parse_args()

    pairs = utils.read_xml(args.test_file)
    x, y, z = fe.pipeline_minimal(pairs)
    
    if args.classifier:
        eval_classifier(args.input, x, y)
    
    if args.regressor:
        eval_regressor(args.input, x, z)
    

    