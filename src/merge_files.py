# -*- coding: utf-8 -*-


'''
Script to merge two RTE files.
'''

import argparse
from xml.etree import cElementTree as ET

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file1', help='First file to merge')
    parser.add_argument('file2', help='Second file to merge')
    parser.add_argument('output', help='Merged file')
    args = parser.parse_args()

    tree1 = ET.parse(args.file1)
    root1 = tree1.getroot()
    last_id = max(int(pair.get('id'))
                  for pair in root1.iter('pair'))

    tree2 = ET.parse(args.file2)
    root2 = tree2.getroot()

    new_id = last_id + 1
    for xml_pair in root2.iter('pair'):
        # change pair ids to avoid repeated ones
        xml_pair.set('id', str(new_id))
        new_id += 1

    root1.extend(root2)
    tree1.write(args.output, 'utf-8', True)

