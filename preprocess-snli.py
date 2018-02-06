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


def read_pairs(path):
    """
    Read pairs from the given path.
    """
    pairs = []
    pair_counter = 1
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # if data['gold_label'] == '-':
            #     # ignore items without a gold label
            #     continue

            sent1 = data['sentence1']
            sent2 = data['sentence2']
            label = data['gold_label']

            pair = ds.Pair(sent1, sent2, pair_counter, label)
            pairs.append(pair)

    return pairs


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
    parser.add_argument('label_dict', help='Dictionary of dependency labels in '
                                           'json format')
    parser.add_argument('output', help='Path to save generated pairs as pickle')
    args = parser.parse_args()

    pairs = read_pairs(args.snli)
    parses = read_parses(args.trees)
    wd = utils.load_vocabulary(args.vocabulary)
    label_dict = utils.load_label_dict(args.label_dict)

    print('%d pairs and %d parses' % (len(pairs), len(parses)))

    for i, pair in enumerate(pairs):
        if pair.entailment == '-':
            pairs[i] = None
            continue

        pair.entailment = utils.map_entailment_string(pair.entailment)

        parse1 = parses[2*i]
        sent1 = ds.Sentence(pair.t, parse1, wd, label_dict)
        pair.annotated_t = sent1

        parse2 = parses[2*i + 1]
        sent2 = ds.Sentence(pair.h, parse2, wd, label_dict)
        pair.annotated_h = sent2

        if i % 10000 == 0:
            print('Read %d pairs' % i)

    pairs = [pair for pair in pairs if pair is not None]
    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f)
