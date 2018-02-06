# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Utility functions
'''

import sys
from six.moves import cPickle
from xml.etree import cElementTree as ET
import logging
from collections import defaultdict
from xml.dom import minidom
import numpy as np
import json
import os

from . import config
from . import datastructures as ds


UNKNOWN = '<unk>'


def print_cli_args():
    """
    Log the command line arguments
    :return:
    """
    args = ' '.join(sys.argv)
    logging.info('The following command line arguments were given: %s' % args)


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


def read_pickled_pairs(path, add_inverted=False,
                       paraphrase_to_entailment=False):
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


def load_vocabulary(path):
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
    if UNKNOWN not in d:
        d[UNKNOWN] = len(d)

    unk_index = d[UNKNOWN]
    word_dict = defaultdict(lambda: unk_index, d)

    return word_dict


def load_pairs(filename):
    '''
    Read the pairs from the given file.

    :param filename: either a TSV or an XML file
    :return: list of pairs
    '''
    lower_filename = filename.lower()
    if lower_filename.endswith('.tsv'):
        return load_tsv(filename)
    elif lower_filename.endswith('.xml'):
        return load_xml(filename)
    else:
        msg = 'Unrecognized file extension (expecting .tsv or .xml): %s'
        msg %= filename
        raise ValueError(msg)


def load_tsv(filename):
    '''
    Read a TSV file with the format, as used in the SICK corpus.

    sentence1 [TAB] sentence2 [TAB] label [TAB] similarity

    :param filename: path to a tsv file
    :return: list of pairs
    '''
    pairs = []
    with open(filename, 'r') as f:
        for line in f:
            sent1, sent2, label, similarity = line.split('\t')
            entailment = map_entailment_string(label)
            similarity = float(similarity)
            pair = ds.Pair(sent1, sent2, None, entailment, similarity)
            pairs.append(pair)

    return pairs


def map_entailment_string(ent_string):
    """
    Map an entailment string (written in a corpus file) to an Entailment enum
    object
    """
    ent_string = ent_string.lower()
    if ent_string in ['yes', 'entailment']:
        entailment = ds.Entailment.entailment
    elif ent_string == 'paraphrase':
        entailment = ds.Entailment.paraphrase
    elif ent_string == 'contradiction':
        entailment = ds.Entailment.contradiction
    else:
        entailment = ds.Entailment.none
    return entailment


def load_xml(filename):
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
            entailment = map_entailment_string(ent_string)
                        
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
        text = f.read()
    words = [word.decode('utf-8', 'backslashreplace')
             for word in text.splitlines()]

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


def load_embeddings_and_dict(embeddings_path, normalize=True):
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
    if UNKNOWN in wd:
        unk_index = wd[UNKNOWN]
    else:
        unk_index = max(wd.values()) + 1
        wd[UNKNOWN] = unk_index
        mean = embeddings.mean()
        std = embeddings.std()
        shape = [1, embeddings.shape[1]]
        unk_vector = np.random.normal(mean, std, shape)
        embeddings = np.concatenate([embeddings, unk_vector])

    wd = defaultdict(lambda: unk_index, wd)

    logging.debug('Embeddings have shape {}'.format(embeddings.shape))
    if normalize:
        embeddings = normalize_embeddings(embeddings)

    return wd, embeddings


def load_embeddings(path, add_vectors=None):
    """
    Load an embedding model from the given path

    :param path: path to a numpy file
    :param add_vectors: number of vectors to add to the embedding matrix. They
        will be generated randomly with mean and stdev from the others.
    :return: numpy array
    """
    embeddings = np.load(path)
    if add_vectors:
        mean = embeddings.mean()
        std = embeddings.std()
        shape = [add_vectors, embeddings.shape[1]]
        new_vectors = np.random.normal(mean, std, shape)
        embeddings = np.concatenate([embeddings, new_vectors])

    return embeddings


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
