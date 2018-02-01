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


def preprocess_pairs(pairs):
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
            pair.annotated_t = ds.Sentence(pair.t, output_t, 'conll')
            pair.annotated_h = ds.Sentence(pair.h, output_h, 'conll')
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)

        # pair.lexical_alignments = utils.find_lexical_alignments(pair)
        # pair.ppdb_alignments = utils.find_ppdb_alignments(pair, max_length=5)
        pairs[i] = pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='XML or TSV file with pairs')
    parser.add_argument('output',
                        help='File to write output file (pickle format)')
    args = parser.parse_args()

    pairs = utils.read_pairs(args.input)
    preprocess_pairs(pairs)
    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f)
