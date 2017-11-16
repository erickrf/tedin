# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Functions for calling external NLP tools.
"""

import urllib
import json
import requests

import config

nlpnet_tagger = None


# def load_nlpnet_tagger(language='pt'):
#     """
#     Load and return the nlpnet POS tagger.
#     Its location in the file system is read from the config module.
#     """
#     # store in global variable to avoid loading times
#     global nlpnet_tagger
#     if nlpnet_tagger is not None:
#         return nlpnet_tagger
#
#     nlpnet_tagger = nlpnet.POSTagger(config.nlpnet_path_pt, language=language)
#     return nlpnet_tagger

def _is_trivial_paraphrase(exp1, exp2):
    """
    Return True if w1 and w2 differ only in gender and/or number

    :param exp1: tuple/list of strings, expression1
    :param exp2: tuple/list of strings, expression2
    :return: boolean
    """
    def strip_suffix(word):
        if word[-2:] == 'os' or word[-2:] == 'as':
            return word[:-2]

        if word[-1] in 'aos':
            return word[:-1]

        return word

    if len(exp1) != len(exp2):
        return False

    for w1, w2 in zip(exp1, exp2):
        w1 = strip_suffix(w1)
        w2 = strip_suffix(w2)
        if len(w1) == 0 or len(w2) == 0:
            if len(w1) == len(w2):
                continue
            else:
                return False

        if w1 != w2 and \
                not (w1[-1] == 'l' and w2[-1] == 'i' and w1[:-1] == w2[:-1]):
            return False

    return True


def load_ppdb(path):
    """
    Load a paraphrase file from Paraphrase Database.

    :param path: path to the file
    :return: a nested dictionary containing transformations.
        each level of the dictionary has one token of the right-hand side of
        the transformation rule mapping to a tuple (transformations, dict):
        ex:
        {'poder': (set(),
                   {'legislativo': (set('legislatura',
                                    {})})
        }
    """
    transformations = {}
    articles = {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'}

    def remove_comma_and_article(expression):
        if len(expression) == 1:
            return expression
        if expression[0] == ',':
            expression = expression[1:]
        if expression[-1] == ',':
            expression = expression[:-1]
        if expression[0] in articles:
            expression = expression[1:]
        return expression

    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            fields = line.split('|||')
            lhs = fields[1].strip().split()
            rhs = fields[2].strip().split()

            lhs = tuple(remove_comma_and_article(lhs))
            rhs = tuple(remove_comma_and_article(rhs))

            # filter out trivial number/gender variations
            if _is_trivial_paraphrase(lhs, rhs):
                continue

            # add rhs to the nested dictionary
            d = transformations
            for token in rhs:
                if token in d:
                    d = d[token]

            if lhs in transformations:
                transformations[lhs].add(rhs)
            else:
                transformations[lhs] = {rhs}

    return transformations


def call_palavras(text):
    """
    Call a webservice to run the parser Palavras

    :param text: the text to be parsed, in unicode.
    :return: the response string from Palavras
    """
    params = {'sentence': text.encode('utf-8')}
    data = urllib.urlencode(params)
    f = urllib.urlopen(config.palavras_endpoint, data)
    response = f.read()
    f.close()

    return response


def call_corenlp(text):
    """
    Call Stanford corenlp, which should be running at the address specified in
    the config module.

    Only a dependency parser and POS tagger are run.

    :param text: text with tokens separated by whitespace
    """
    properties = {'tokenize.whitespace': 'true',
                  'annotators': 'tokenize,ssplit,pos,depparse',
                  'depparse.model': config.corenlp_depparse_path,
                  'pos.model': config.corenlp_pos_path,
                  'outputFormat': 'conllu'}

    # use json dumps function to convert the nested dictionary to a string
    properties_val = json.dumps(properties)
    params = {'properties': properties_val}

    # we encode the URL params using urllib because we need a URL with GET parameters
    # even though we are making a POST request. The POST data is the text itself.
    encoded_params = urllib.urlencode(params)
    url = '{url}:{port}/?{params}'.format(url=config.corenlp_url,
                                          port=config.corenlp_port,
                                          params=encoded_params)

    headers = {'Content-Type': 'text/plain;charset=utf-8'}
    response = requests.post(url, text.encode('utf-8'), headers=headers)
    response.encoding = 'utf-8'

    # bug: \0 character appears in the response
    output = response.text.replace('\0', '')

    return output
