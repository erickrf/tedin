# -*- coding: utf-8 -*-

"""
Preprocess the pairs with dependency trees.

This is useful when preprocessing takes time.
"""

import argparse
import traceback
import logging
from six.moves import cPickle

from infernal import config
from infernal import tokenizer
from infernal import utils
from infernal import external
from infernal import datastructures as ds
from infernal import openwordnetpt as own
from infernal import ppdb


def preprocess_pairs(pairs):
    """
    Preprocess the pairs in-place so we can extract features later on.

    :param pairs: list of `Pair` objects
    """
    parser_path = config.corenlp_depparse_path
    pos_path = config.corenlp_pos_path

    for i, pair in enumerate(pairs):
        tokens_t = tokenizer.tokenize(pair.t)
        tokens_h = tokenizer.tokenize(pair.h)

        output_t = external.call_corenlp(' '.join(tokens_t), parser_path,
                                         pos_path)
        output_h = external.call_corenlp(' '.join(tokens_h), parser_path,
                                         pos_path)

        # output_t = external.call_corenlp(pair.annotated_t)
        # output_h = external.call_corenlp(pair.annotated_h)

        try:
            pair.annotated_t = ds.Sentence(output_t)
            pair.annotated_h = ds.Sentence(output_h)
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)
            raise

        pair.find_lexical_alignments()
        # pair.find_ppdb_alignments(max_length=5)

    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='XML or TSV file with pairs')
    parser.add_argument('output',
                        help='File to write output file (pickle format)')
    args = parser.parse_args()

    own.load_wordnet(config.ownpt_path)
    # ppdb.load_ppdb(config.ppdb_path)

    pairs = utils.load_pairs(args.input)
    pairs = preprocess_pairs(pairs)

    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f, -1)
