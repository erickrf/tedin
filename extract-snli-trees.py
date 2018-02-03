# -*- coding: utf-8 -*-

"""
Extract dependency trees from SNLI and save them in a file
"""

from __future__ import division, print_function, unicode_literals

import argparse
import json


def write_lines(lines, path):
    with open(path, 'w') as f:
        text = '\n'.join(lines)
        f.write(text)


def extract_trees(input_path, output_path):
    pair_counter = 0
    file_counter = 1
    lines = []

    with open(input_path, 'r') as fin:
        for line in fin:
            data = json.loads(line)
            s1 = data['sentence1_parse']
            s2 = data['sentence2_parse']

            lines.append(s1)
            lines.append(s2)

            pair_counter += 1
            if pair_counter == 10000:
                path = output_path.replace('.txt', '-%d.txt' % file_counter)
                write_lines(lines, path)

                file_counter += 1
                pair_counter = 0
                lines = []

            # fout.write(s1)
            # fout.write('\n')
            # fout.write(s2)
            # fout.write('\n')

    if pair_counter > 0:
        path = output_path.replace('.txt', '-%d.txt' % file_counter)
        write_lines(lines, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snli', help='SNLI file')
    parser.add_argument('output', help='File to write trees to')
    args = parser.parse_args()

    extract_trees(args.snli, args.output)
