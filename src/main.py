# -*- coding: utf-8 -*-

u'''Script para ler XML do AVE e pegar apenas os
pares que apresentam relação de acarretamento.'''

import argparse
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Arquivo XML do AVE')
    parser.add_argument('output', help=u'Arquivo para escrever a saída')
    args = parser.parse_args()
    
    tree = ET.parse(args.input)
    root = tree.getroot()
    
    # o elemento root é ave-corpus, que tem elementos pair como filhos diretos
    # queremos pegar os positivos, i.e., com atributo value="YES"
    pairs = root.findall('pair[@value="YES"]')
    
    # insere um contador para facilitar a visualização
    for num, pair in enumerate(pairs):
        pair.set('num', str(num + 1))
        pairs[num] = pair
    
    # agora, cria uma nova árvore XML somente com os pares extraídos
    new_root = ET.Element('ave-corpus')
    new_root.extend(pairs)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(args.output, 'UTF-8')
