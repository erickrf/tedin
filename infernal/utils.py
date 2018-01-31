# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Utility functions
'''

import sys
import re
from six.moves import cPickle
from xml.etree import cElementTree as ET
import logging
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from xml.dom import minidom
import numpy as np
import nltk
import json
import os

from . import config
from . import datastructures as ds
from . import openwordnetpt as own
from . import ppdb


content_word_tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PNOUN'}


class EmbeddingDictionary(object):
    '''
    Class for storing a word dictionary to embeddings. It treats special
    cases of OOV words.
    '''
    def __init__(self, wd, embeddings):
        '''
        Create a new EmbeddingDictionary.

        :param wd: path to vocabulary
        :param embeddings: path to 2-d numpy array
        '''
        logging.info('Reading embeddings...')

        # wd should include OOV treatment
        self.wd = read_vocabulary(wd)
        self.embeddings = np.load(embeddings)

    def __getitem__(self, item):
        return self.embeddings[self.wd[item]]

    def get_oov_vector(self):
        return self.embeddings[-1]

    def set_oov_vector(self, vector):
        self.embeddings[-1] = vector

    def get_sentence_embeddings(self, sentence):
        '''
        Return an array with the embeddings of each token in the sentence

        :param sentence: ds.Sentence
        :return: numpy array (num_tokens, embedding_size)
        :rtype: np.ndarray
        '''
        indices = [self.wd[token.text.lower()]
                   for token in sentence.tokens]
        return self.embeddings[indices]


def print_cli_args():
    """
    Log the command line arguments
    :return:
    """
    args = ' '.join(sys.argv)
    logging.info('The following command line arguments were given: %s' % args)


def tokenize_sentence(text, change_quotes=True, change_digits=False):
    '''
    Tokenize the given sentence in Portuguese. The tokenization is done in
    conformity with Universal Treebanks (at least it attempts so).

    :param text: text to be tokenized, as a string
    :param change_quotes: if True, change different kinds of quotation marks to "
    :param change_digits: if True, replaces all digits with 9.
    '''
    if change_digits:
        text = re.sub(r'\d', '9', text)
    
    if change_quotes:
        text = text.replace('“', '"').replace('”', '"')
    
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d+(?:[.,]\d+)*(?:[.,]\d+)|       # numbers in format 999.999.999,99999
    \.{3,}|                           # ellipsis or sequences of dots
    \w+(?:\.(?!\.|$))?|               # words with numbers (including hours as 12h30), 
                                      # followed by a single dot but not at the end of sentence
    \d+:\d+|                          # time and proportions
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    \$|                               # currency sign
    (?:[\#@]\w+])|                    # Hashtags and twitter user names
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)
    
    return tokenizer.tokenize(text)


def nested_list_to_array(sequences, dtype=np.int32, dim3=None):
    """
    Create a numpy array with the content of sequences.

    In case of sublists with different sizes, the array is padded with zeros.

    If the given sequences are a list of lists, a 2d array is created and dim3
    should be None. In case of a 3-level list, the third level sublists must
    always have the same size and be provided in dim3.

    :param sequences: a list of sublists or 3-level lists. In the latter case,
        the third level sublists must always have the same size.
    :param dtype: type of the numpy array
    :return: a tuple (2d array, 1d array) with the data and the sequence sizes
    """
    if len(sequences) == 0:
        raise ValueError('Empty sequence')

    sizes = np.array([len(seq) for seq in sequences], np.int32)

    if dim3 is None:
        shape = [len(sequences), sizes.max()]
    else:
        shape = [len(sequences), sizes.max(), dim3]

    array = np.zeros(shape, dtype)

    for i, seq in enumerate(sequences):
        if len(seq):
            array[i, :sizes[i]] = seq

    return array, sizes


def load_label_dict(path):
    """
    Load the label dictionary from a json file.
    """
    with open(path, 'r') as f:
        d = json.load(f)

    return d


def assign_word_indices(pairs, wd, lower=True):
    """
    Assign each token in the sentences of pairs their embedding index.

    Changes are in-place.

    This is done here instead of in a pre-processing stage to allow for
    different embedding models with different vocabularies.

    :param pairs: list of Pair objects
    :param wd: dictionary mapping strings to ints
    :param lower: whether to lowercase tokens before indexing
    """
    def get_index(token):
        if lower:
            return wd[token.lower()]
        return wd[token]

    for pair in pairs:
        for token in pair.annotated_t.tokens:
            token.index = get_index(token.text)

        for token in pair.annotated_h.tokens:
            token.index = get_index(token.text)


def load_stopwords():
    """
    Return a set of stopwords
    """
    return set(nltk.corpus.stopwords.words('portuguese'))


def find_ppdb_alignments(pair, max_length):
    """
    Find lexical and phrasal alignments in the pair according to transformation
    rules from the paraphrase database.

    :param pair: Pair
    :param max_length: maximum length of the left-hand side (in number of
        tokens)
    """
    tokens_t = pair.annotated_t.tokens
    tokens_h = pair.annotated_h.tokens
    token_texts_t = [token.text.lower() for token in tokens_t]
    token_texts_h = [token.text.lower() for token in tokens_h]
    alignments = []

    ppdb.load_ppdb(config.ppdb_path)

    for i, token in enumerate(tokens_t):
        # check the maximum length that makes sense to search for
        # (i.e., so it doesn't go past sentence end)
        max_possible_length = min(len(tokens_t) - i, max_length)
        for length in range(1, max_possible_length):
            if length == 1 and token.pos not in content_word_tags:
                continue

            lhs = [token for token in token_texts_t[i:i + length]]
            rhs_rules = ppdb.get_rhs(lhs)
            if not rhs_rules:
                continue

            # now get the token objects, instead of just their text
            lhs = tokens_t[i:i + length]

            for rule in rhs_rules:
                index = ppdb.search(token_texts_h, rule)
                if index == -1:
                    continue
                alignment = (lhs, tokens_h[index:index + len(rule)])
                alignments.append(alignment)

    return alignments


def filter_words_by_pos(tokens, tags=None):
    """
    Filter out words based on their POS tags.

    If no set of tags is provided, a default of content tags is used:
    {'NOUN', 'VERB', 'ADJ', 'ADV', 'PNOUN'}

    :param tokens: list of datastructures.Token objects
    :param tags: optional set of allowed tags
    :return: list of the tokens having the allowed tokens
    """
    if tags is None:
        tags = content_word_tags

    return [token for token in tokens if token.pos in tags]


def find_lexical_alignments(pair):
    '''
    Find the lexical alignments in the pair.
    
    Lexical alignments are simply two equal or synonym words.
    
    :type pair: datastructures.Pair
    :return: list with the (Token, Token) aligned tuples
    '''
    # pronouns aren't content words, but let's pretend they are
    content_word_tags = {'NOUN', 'VERB', 'PRON', 'ADJ', 'ADV', 'PNOUN'}
    content_words_t = [token for token in filter_words_by_pos(
                            pair.annotated_t.tokens, content_word_tags)
                       # own-pt lists ser and ter as synonyms
                       if token.lemma not in ['ser', 'ter']]

    content_words_h = [token for token in filter_words_by_pos(
                            pair.annotated_h.tokens, content_word_tags)
                       if token.lemma not in ['ser', 'ter']]
    
    lexical_alignments = []
    
    for token in pair.annotated_t.tokens:
        token.aligned_to = []
    for token in pair.annotated_h.tokens:
        token.aligned_to = []

    own.load_wordnet(config.ownpt_path)
    for token_t in content_words_t:
        nominalizations_t = own.find_nominalizations(token_t.lemma)

        for token_h in content_words_h:
            aligned = False
            if token_t.lemma == token_h.lemma:
                aligned = True
            elif own.are_synonyms(token_t.lemma, token_h.lemma):
                aligned = True
            elif token_h.lemma in nominalizations_t:
                aligned = True
            elif token_t.lemma in own.find_nominalizations(token_h.lemma):
                aligned = True

            if aligned:
                lexical_alignments.append((token_t, token_h))
                token_t.aligned_to.append(token_h)
                token_h.aligned_to.append(token_t)

    return lexical_alignments


def extract_classes(pairs):
    '''
    Extract the class infomartion (paraphrase, entailment, none, contradiction)
    from the pairs. 
    
    :return: a numpy array with values from 0 to num_classes - 1
    '''
    classes = np.array([pair.entailment.value for pair in pairs])
    return classes


def extract_similarities(pairs):
    '''
    Extract the similarity value from the pairs.
    
    :return: a numpy array
    '''
    z = np.array([p.similarity for p in pairs])
    return z


def read_pairs(path, add_inverted=False, paraphrase_to_entailment=False):
    '''
    Load pickled pairs from the given path.

    :param path: pickle file path
    :param add_inverted: augment the set with the inverted pairs
    :param paraphrase_to_entailment: change paraphrase class to
        entailment
    :return: list of pairs
    '''
    logging.info('Reading pairs...')
    with open(path, 'rb') as f:
        pairs = cPickle.load(f)
    logging.info('Read %d pairs' % len(pairs))

    if add_inverted:
        extra_pairs = []
        for pair in pairs:
            if pair.entailment == ds.Entailment.paraphrase:
                ent_value = ds.Entailment.paraphrase
            else:
                # inverting None and Entailment classes yields None
                ent_value = ds.Entailment.none

            inverted = pair.inverted_pair(ent_value)
            extra_pairs.append(inverted)

        pairs.extend(extra_pairs)
        logging.debug('%d total pairs after adding inverted ones' % len(pairs))

    count = 0
    if paraphrase_to_entailment:
        for pair in pairs:
            if pair.entailment == ds.Entailment.paraphrase:
                count += 1
                pair.entailment = ds.Entailment.entailment

        logging.debug('Changed %d paraphrases to entailment' % count)

    return pairs


def combine_paraphrase_predictions(predictions1, predictions2):
    '''
    Combine two sequences of binary predictions (entailment or none)
    into one sequence of three way predictions (entailment, none or paraphrase).
    For any pair, if the second prediction is entailment and the first one
    is none, the result will be none.

    :param predictions1: 1-d numpy array
    :param predictions2: 1-d numpy array
    :return: 1-d numpy array
    '''
    ent_value = ds.Entailment.entailment.value
    idx_paraphrases = np.logical_and(predictions1 == ent_value,
                                     predictions2 == ent_value)
    combined = predictions1.copy()
    combined[idx_paraphrases] = ds.Entailment.paraphrase.value
    return combined


def read_vocabulary(path):
    """
    Read a file with the vocabulary corresponding to an embedding model.

    :param path: path to the file (should be UTF-8!)
    :return: a python dictionary mapping words to indices in the
        embedding matrix
    """
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')

    words = text.splitlines()
    values = range(len(words))
    d = dict(zip(words, values))
    unk_index = d['<unk>']
    word_dict = defaultdict(lambda: unk_index, d)

    return word_dict


def read_xml(filename):
    '''
    Read an RTE XML file and return a list of Pair objects.
    '''
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        
        # the entailment relation is expressed differently in some versions
        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()
            
            if ent_string in ['yes', 'entailment']:
                entailment = ds.Entailment.entailment
            elif ent_string == 'paraphrase':
                entailment = ds.Entailment.paraphrase
            elif ent_string == 'contradiction':
                entailment = ds.Entailment.contradiction
            else:
                entailment = ds.Entailment.none
                        
        elif 'value' in attribs:
            if attribs['value'].lower() == 'true':
                entailment = ds.Entailment.entailment
            else:
                entailment = ds.Entailment.none
            
        if 'similarity' in attribs:
            similarity = float(attribs['similarity']) 
        else:
            similarity = None
        
        id_ = int(attribs['id'])
        pair = ds.Pair(t, h, id_, entailment, similarity)
        pairs.append(pair)
    
    return pairs


def write_rte_file(filename, pairs, **attribs):
    '''
    Write an XML file containing the given RTE pairs.
    
    :param pairs: list of Pair objects
    '''
    root = ET.Element('entailment-corpus')
    for i, pair in enumerate(pairs, 1):
        xml_attribs = {'id':str(i)}
        
        # add any other attributes supplied in the function call or the pair
        xml_attribs.update(attribs)
        xml_attribs.update(pair.attribs)
        
        xml_pair = ET.SubElement(root, 'pair', xml_attribs)
        xml_t = ET.SubElement(xml_pair, 't', pair.t_attribs)
        xml_h = ET.SubElement(xml_pair, 'h', pair.h_attribs)
        xml_t.text = pair.t.strip()
        xml_h.text = pair.h.strip()
        
    # produz XML com formatação legível (pretty print)
    xml_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(xml_string)
    with open(filename, 'wb') as f:
        f.write(reparsed.toprettyxml('    ', '\n', 'utf-8'))


def train_classifier(x, y):
    '''
    Train and return a classifier with the supplied data
    '''
    classifier = config.classifier_class(class_weight='auto')
    classifier.fit(x, y)
    
    return classifier


def train_regressor(x, y):
    '''
    Train and return a regression model (for similarity) with the supplied data.
    '''
    regressor = config.regressor_class()
    regressor.fit(x, y)
    
    return regressor


def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            words.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)

    embeddings = np.array(vectors, dtype=np.float32)

    return words, embeddings


def load_binary_embeddings(embeddings_path, vocabulary_path):
    """
    Load any embedding model in numpy format, and a corresponding
    vocabulary with one word per line.

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with words
    :return: a tuple (wordlist, array)
    """
    vectors = np.load(embeddings_path)

    with open(vocabulary_path, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')
    words = text.splitlines()

    return words, vectors


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def load_embeddings(embeddings_path, normalize=True):
    """
    Load and return an embedding model in either text format or
    numpy binary format. If the file extension is .txt, text format is
    assumed. If it is .npy, a corresponding vocabulary file is sought with
    the same name and .txt.

    :param embeddings_path: path to embeddings file
    :param normalize: whether to normalize embeddings
    :return: a tuple (defaultdict, array)
    """

    logging.debug('Loading embeddings')
    base_path, ext = os.path.splitext(embeddings_path)
    if ext.lower() == '.txt':
        wordlist, embeddings = load_text_embeddings(embeddings_path)
    else:
        vocabulary_path = base_path + '.txt'
        wordlist, embeddings = load_binary_embeddings(embeddings_path,
                                                      vocabulary_path)

    wd = {word: ind for ind, word in enumerate(wordlist)}
    unk_index = wd['<unk>']
    wd = defaultdict(lambda: unk_index, wd)

    logging.debug('Embeddings have shape {}'.format(embeddings.shape))
    if normalize:
        embeddings = normalize_embeddings(embeddings)

    return wd, embeddings


def get_logger(name='logger'):
    """
    Setup and return a simple logger.
    :return:
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.propagate = False

    return logger
