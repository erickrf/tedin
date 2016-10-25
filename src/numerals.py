# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Conversion from number strings to ints.
'''

import logging
import re

values = {'zero': 0,
          'um': 1,
          'uma': 2,
          'dois': 2,
          'duas': 2,
          'três': 3,
          'quatro': 4,
          'cinco': 5,
          'seis': 6,
          'sete': 7,
          'oito': 8,
          'nove': 9,
          'dez': 10,
          'onze': 11,
          'doze': 12,
          'treze': 13,
          'quatorze': 14,
          'catorze': 14,
          'quinze': 15,
          'dezesseis': 16,
          'dezessete': 17,
          'dezoito': 18,
          'dezenove': 19,
          'vinte': 20,
          'trinta': 30,
          'quarenta': 40,
          'cinquenta': 50,
          'sessenta': 60,
          'setenta': 70,
          'oitenta': 80,
          'noventa': 90,
          'cem': 100,
          'duzentos': 200,
          'trezentos': 300,
          'quatrocentos': 400,
          'quinhentos': 500,
          'seiscentos': 600,
          'setecentos': 700,
          'oitocentos': 800,
          'novecentos': 900,
          'mil': 1000,
}


def get_number(token):
    '''
    Return a number representation in this token. The value might be the
    content of this token itself or composed with its dependents.
    
    If it's not possible, raise a ValueError.
    '''
    text = token.text
    if re.match(r'[.,\d]+$', text):
        en_format = text.replace('.', '').replace(',', '.')
        return float(en_format)
    
    if len(token.dependents) > 0:
        dependents_str = ', '.join(d.text for d in token.dependents)
        msg = 'Token {} has dependents: {}'.format(text, dependents_str)
        logging.debug(msg.encode('utf-8'))
    
    try:
        return values[text.lower()]
    except KeyError:
        msg = "Can't convert this number to digits: {}".format(text)
        logging.debug(msg.encode('utf-8'))
        return text

