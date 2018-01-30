# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Functions for calling external NLP tools.
"""

import json
import requests
from six.moves import urllib

import config

nlpnet_tagger = None


def call_palavras(text):
    """
    Call a webservice to run the parser Palavras

    :param text: the text to be parsed, in unicode.
    :return: the response string from Palavras
    """
    params = {'sentence': text.encode('utf-8')}
    data = urllib.parse.urlencode(params)
    f = urllib.request.urlopen(config.palavras_endpoint, data)
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
    encoded_params = urllib.parse.urlencode(params)
    url = '{url}:{port}/?{params}'.format(url=config.corenlp_url,
                                          port=config.corenlp_port,
                                          params=encoded_params)

    headers = {'Content-Type': 'text/plain;charset=utf-8'}
    response = requests.post(url, text.encode('utf-8'), headers=headers)
    response.encoding = 'utf-8'

    # bug: \0 character appears in the response
    output = response.text.replace('\0', '')

    return output
