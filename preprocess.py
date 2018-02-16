# -*- coding: utf-8 -*-

"""
Preprocess the pairs with dependency trees.

This is useful when preprocessing takes time.
"""

import argparse
import traceback
import logging
from six.moves import cPickle

# from . import tokenizer
from infernal import utils
from infernal import external
from infernal import datastructures as ds


def preprocess_pairs(pairs, wd, dep_dict, lower):
    """
    Preprocess the pairs in-place so we can extract features later on.

    :param pairs: list of `Pair` objects
    """
    new_pairs = []
    for i, pair in enumerate(pairs):
        # tokens_t = tokenizer.tokenize(pair.t)
        # tokens_h = tokenizer.tokenize(pair.h)

        # output_t = external.call_corenlp(' '.join(tokens_t))
        # output_h = external.call_corenlp(' '.join(tokens_h))

        output_t = external.call_corenlp(pair.annotated_t)
        output_h = external.call_corenlp(pair.annotated_h)

        try:
            sent1 = ds.Sentence(output_t, wd, dep_dict, lower)
            sent2 = ds.Sentence(output_h, wd, dep_dict, lower)
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)
            raise

        new_pair = ds.Pair(sent1, sent2, pair.label)
        new_pairs.append(new_pair)

    return new_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='XML or TSV file with pairs')
    parser.add_argument('vocabulary', help='Vocabulary file (corresponding to '
                                           'an embedding matrix)')
    parser.add_argument('dep_dict', help='Dictionary of dependency labels in '
                                         'json format')
    parser.add_argument('--lower', action='store_true',
                        help='Lowercase tokens before indexing')
    parser.add_argument('output',
                        help='File to write output file (pickle format)')
    args = parser.parse_args()

    wd = utils.load_vocabulary(args.vocabulary)
    dep_dict = utils.load_label_dict(args.dep_dict)
    pairs = utils.load_pairs(args.input)
    pairs = preprocess_pairs(pairs, wd, dep_dict, args.lower)

    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f, -1)
