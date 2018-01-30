# -*- encoding: utf-8 -*-

import os
import re
import argparse

import utils
from datastructures import Pair


def get_value(pattern, content ):
    '''
    Auxilar function to get value
    '''
    value = re.findall( pattern, content )
    if len(value) == 0:
        return ''

    return value[0].split('.')[ 0 ]
    

def read_document(cst_directory, document_prefix):
    
    text_dir_path = os.path.join(cst_directory, '..', 'Textos-fonte segmentados')
    filenames = os.listdir(text_dir_path)
    
    for filename in filenames:
        if filename.startswith(document_prefix):
            break
    
    path = os.path.join(text_dir_path, filename)
    with open(path) as f:
        text = unicode(f.read(), 'utf-8')
    
    return text.splitlines() 


def read_cst_news( cst_news_directory ):
    '''
    Read the content of the CST news corpus and return a list of relations.
    Each relation is a tuple in the format (sentence1, sentence2, relation name).
    '''
    relations = []
    for cluster_dir in os.listdir( cst_news_directory ):
        
        if not cluster_dir.startswith('C'):
            continue
        
        cluster_path = os.path.join(cst_news_directory, cluster_dir)
        relations.extend(read_cluster(cluster_path))
    
    return relations

def read_cluster(cluster_path):
    documents = {}
    cst_path = os.path.join(cluster_path, 'CST')
    data = []
    for filename in os.listdir(cst_path):
        if not filename.endswith('.cst'):
            continue
        
        with open(os.path.join(cst_path, filename)) as cstfile:
            text = cstfile.read()
        
        relations = re.findall('<R S[^>]+>[\n\r \t]*<RELATION[^/]+/>',
                               text)
        for relation in relations:
            sdoc = get_value('(?<=SDID=\")[^\"]+', relation )
            ssent = get_value('(?<=SSENT=\")[^\"]+', relation )

            if not sdoc in documents:
                documents[sdoc] = read_document(cst_path, sdoc)

            tdoc = get_value('(?<=TDID=\")[^\"]+', relation )
            tsent = get_value('(?<=TSENT=\")[^\"]+', relation )
            if not tdoc in documents:
                documents[tdoc] = read_document(cst_path, tdoc)

            kind =  re.findall(
                '(?<=RELATION TYPE=\")[^\"]+',
                relation )[ 0 ].capitalize()

            #The CSTPArser starts the index at 1.                
            ssent = int( ssent ) -1
            tsent = int( tsent ) -1
            
            sent1 = documents[sdoc][ssent]
            sent2 = documents[tdoc][tsent]
            
            data.append((sent1, sent2, kind))
        
    return data

def create_rte_file(relations, filename, max_t_size, max_h_size):
    '''
    Create an RTE XML file from the given CST relations
    '''
    useful_relations = set(['Subsumption', 'Overlap', 'Equivalence', 'Contradiction'])
    used_pairs = set()
    pairs = []
    
    for sent1, sent2, relation in relations:
        if relation not in useful_relations:
            continue
        
        tokens1 = utils.tokenize_sentence(sent1, False)
        if max_t_size > 0 and len(tokens1) > max_t_size:
            continue
        
        tokens2 = utils.tokenize_sentence(sent2, False)
        if max_h_size > 0 and len(tokens2) > max_h_size:
            continue
        
        if (sent1, sent2) in used_pairs:
            continue
        
        used_pairs.add((sent1, sent2))
        p = Pair(sent1, sent2, 'UNKNOWN', cst=relation)
        pairs.append(p)
        
    utils.write_rte_file(filename, pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cst_directory', help='Directory containing CST news corpus')
    parser.add_argument('rte_file', help='Name of the XML file to be generated')
    parser.add_argument('--max-t-size', help='Maximum T size in tokens', 
                        dest='max_t_size', default=0, type=int)
    parser.add_argument('--max-h-size', help='Maximum H size in tokens', 
                        dest='max_h_size', default=0, type=int)
    args = parser.parse_args()
    
    relations = read_cst_news(args.cst_directory)
    create_rte_file(relations, args.rte_file, args.max_t_size, args.max_h_size)
    
