# -*- coding: utf-8 -*-

'''
Functions to read the OpenWordnetPT from RDF files and provide
access to it.
'''

import rdflib

import config

lexical_form_predicate = rdflib.URIRef(u'https://w3id.org/own-pt/wn30/schema/lexicalForm')
word_type = rdflib.URIRef('https://w3id.org/own-pt/wn30/schema/Word')
word_pred = rdflib.URIRef('https://w3id.org/own-pt/wn30/schema/word')
word_sense_type = rdflib.URIRef('https://w3id.org/own-pt/wn30/schema/WordSense')
type_pred = rdflib.RDF.type
contains_sense_pred = rdflib.URIRef('https://w3id.org/own-pt/wn30/schema/containsWordSense')

def read_graph():
    '''
    Read the graph and return it
    '''
    g = rdflib.Graph()
    g.parse(config.ownpt_path, format='nt')
    
    return g

def find_synsets(graph, word):
    '''
    Find and return all synsets containing the given word in the given graph.
    
    :type word: unicode string
    '''
    all_synsets = set()
    word_literal = rdflib.Literal(word, 'pt')
    
    # a word node in the graph has the property lexicalForm 
    # connecting it to the actual text string
    word_node = graph.value(None, lexical_form_predicate, word_literal)
    
    # word nodes are linked to word sense nodes
    word_senses_iter = graph.subjects(word_pred, word_node)
    
    for word_sense in word_senses_iter:
        type_ = graph.value(word_sense, type_pred)
        assert type_ == word_sense_type
        
        synsets_iter = graph.subjects(contains_sense_pred, word_sense)
        synsets = list(synsets_iter)
        all_synsets.update(synsets)
    
    return all_synsets
        
    