# -*- coding: utf-8 -*-

"""
Preprocess the pairs with dependency trees.

This is useful when preprocessing takes time.
"""

import argparse
import traceback
import logging

import tokenizer
import cPickle
import utils
import config
import external
import datastructures as ds


def preprocess_pairs(pairs, parser, binarize):
    """
    Preprocess the pairs so we can extract features later on.

    :param pairs: list of `Pair` objects
    :param parser: which parser to call. Currently supports
        'corenlp', 'malt' and 'palavras'
    :param binarize: Paraphrases pairs turned into two entailment pairs.

    """
    if parser == 'corenlp':
        parser_function = external.call_corenlp
        parser_format = 'conll'
    elif parser == 'palavras':
        parser_function = external.call_palavras
        parser_format = 'palavras'
    elif parser == 'malt':
        parser_function = external.call_malt
        parser_format = 'conll'
    else:
        raise ValueError('Unknown parser: %s' % parser)

    reversed_paraphrases = []
    for i, pair in enumerate(pairs):
        tokens_t = tokenizer.tokenize(pair.t)
        tokens_h = tokenizer.tokenize(pair.h)

        output_t = parser_function(' '.join(tokens_t))
        output_h = parser_function(' '.join(tokens_h))

        try:
            pair.annotated_t = ds.Sentence(output_t, parser_format)
            pair.annotated_h = ds.Sentence(output_h, parser_format)
        except ValueError as e:
            tb = traceback.format_exc()
            logging.error('Error reading parser output:', e)
            logging.error(tb)

        utils.find_lexical_alignments(pair)
        if binarize and pair.entailment == ds.Entailment.paraphrase:
            pair.entailment = ds.Entailment.entailment
            reversed_paraphrases.append(pair.inverted_pair())

        pairs[i] = pair

    pairs.extend(reversed_paraphrases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='XML file with pairs')
    parser.add_argument('output', help='File to write output file (pickle format)')
    parser.add_argument('-b', help='Binarize (turn paraphrases into two entailments)',
                        action='store_true', dest='binarize')
    args = parser.parse_args()

    pairs = utils.read_xml(args.input)
    preprocess_pairs(pairs, config.parser, args.binarize)
    with open(args.output, 'wb') as f:
        cPickle.dump(pairs, f)

