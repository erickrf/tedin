# -*- coding: utf-8 -*-

"""
Script to clean the output produced by the LSTM code
in the notebooks, leaving only numbers (losses)
"""

import re
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', help='Input file')
parser.add_argument('output', help='Output file')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    text = unicode(f.read(), 'utf-8')

text = re.sub(r'Epoch.*', '', text)
text = text.replace('\n\n', '\n')
text = re.sub(r'Loss: ([\d.]+)', r'\1', text)

with open(args.output, 'wb') as f:
    f.write(text.encode('utf-8'))
