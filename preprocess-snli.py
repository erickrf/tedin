# -*- coding: utf-8 -*-

"""
Read the SNLI corpus and pre-processed dependency trees to create and serialize
pair objects.
"""

from __future__ import division, print_function, unicode_literals

import argparse
from six.moves import cPickle
import json

from infernal import datastructures as ds
from infernal import utils


def read_labels(path):
    """
    Read pairs labels from the given path.
    """
    labels = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # if data['gold_label'] == '-':
            #     # ignore items without a gold label
            #     continue

            labels.append(data['gold_label'])

    return labels


def read_parses(path):
    """
    Read all parse trees in a file, separated by empty lines
    """
    with open(path, 'r') as f:
        text = f.read()

    parses = [parse for parse in text.split('\n\n')
              if len(parse.strip()) > 0]
    return parses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snli', help='SNLI corpus file in jsonl format')
    parser.add_argument('trees', help='File with CONLLU dependency trees')
    parser.add_argument('vocabulary', help='Vocabulary file (corresponding to '
                                           'an embedding matrix)')
    parser.add_argument('dep_dict', help='Dictionary of dependency labels in '
                                         'json format')
    parser.add_argument('output', help='Path to save generated pairs as pickle')
    args = parser.parse_args()

    labels = read_labels(args.snli)
    parses = read_parses(args.trees)
    wd = utils.load_vocabulary(args.vocabulary)
    dep_dict = utils.load_label_dict(args.dep_dict)

    print('%d pairs and %d parses' % (len(labels), len(parses)))

    pairs = []
    for i, label in enumerate(labels):
        if label == '-':
            continue

        label = utils.map_entailment_string(label)

        parse1 = parses[2*i]
        sent1 = ds.Sentence(parse1, wd, dep_dict)

        parse2 = parses[2*i + 1]
        sent2 = ds.Sentence(parse2, wd, dep_dict)

        pair = ds.Pair(sent1, sent2, label)
        pairs.append(pair)

        if i % 10000 == 0 or i == len(labels) - 1:
            print('Read %d pairs' % i)

    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f, -1)
