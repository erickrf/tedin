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


def preprocess_pairs(pairs, wd, label_dict, lower):
    """
    Preprocess the pairs in-place so we can extract features later on.

    :param pairs: list of `Pair` objects
    """
    for i, pair in enumerate(pairs):
        # tokens_t = tokenizer.tokenize(pair.t)
        # tokens_h = tokenizer.tokenize(pair.h)

        # output_t = external.call_corenlp(' '.join(tokens_t))
        # output_h = external.call_corenlp(' '.join(tokens_h))

        output_t = external.call_corenlp(pair.t)
        output_h = external.call_corenlp(pair.h)

        try:
            sent1 = ds.Sentence(pair.t, output_t, wd, label_dict, lower)
            sent2 = ds.Sentence(pair.h, output_h, wd, label_dict, lower)
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)
            raise

        pair.annotated_t = sent1
        pair.annotated_h = sent2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='XML or TSV file with pairs')
    parser.add_argument('vocabulary', help='Vocabulary file (corresponding to '
                                           'an embedding matrix)')
    parser.add_argument('label_dict', help='Dictionary of dependency labels in '
                                           'json format')
    parser.add_argument('--lower', action='store_true',
                        help='Lowercase tokens before indexing')
    parser.add_argument('output',
                        help='File to write output file (pickle format)')
    args = parser.parse_args()

    wd = utils.load_vocabulary(args.vocabulary)
    label_dict = utils.load_label_dict(args.label_dict)
    pairs = utils.load_pairs(args.input)
    preprocess_pairs(pairs, wd, label_dict, args.lower)

    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f)
