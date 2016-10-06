# -*- coding: utf-8 -*-

'''
Functions to read the OpenWordnetPT from RDF files and provide
access to it.
'''

import rdflib

import config

ownns = rdflib.Namespace('https://w3id.org/own-pt/wn30/schema/')
lexical_form_predicate = rdflib.URIRef(u'https://w3id.org/own-pt/wn30/schema/lexicalForm')
word_type = ownns['Word']
word_pred = ownns['word']
word_sense_type = ownns['WordSense']
type_pred = rdflib.RDF.type
contains_sense_pred = ownns['containsWordSense']

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
    :return: a set of synsets (rdflib objects)
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
        
def get_synset_words(graph, synset):
    '''
    Return the words of a synset
    :param graph:
    :param synset:
    :return: a list of strings
    '''
    words = []

    # a synset have many word senses
    # each word sense has a Word object and each Word has a lexical form
    senses = graph.objects(synset, contains_sense_pred)
    for sense in senses:
        word_node = g.value(sense, word_pred, any=False)
        word_literal = graph.value(word_node, lexical_form_predicate, any=False)
        words.append(word_literal.toPython())

    return words
