# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
This is the global configuration file. It contains configurations for resources
and external tools that are shared by all configurations.
'''

# ==============
# Parsing config
# ==============

# this must be one of corenlp, palavras or malt
parser = 'corenlp'

# corenlp_url = r'http://143.107.183.175'
# corenlp_port = 13388
corenlp_url = 'http://localhost'
corenlp_port = 9000

# path to the corenlp models inside the server
corenlp_depparse_path = r'models/pt-br/dep-parser'
corenlp_pos_path = 'models/pt-br/pos-tagger.dat'
# corenlp_depparse_path = None
# corenlp_pos_path = None

palavras_endpoint = 'http://143.107.183.175:12680/services/service_palavras_flat.php'
# URL dentro da rede NILC
#palavras_endpoint = 'http://10.11.14.126/services/service_palavras_flat.php'

malt_jar = r'data/malt/maltparser-1.8.1/maltparser-1.8.1.jar'
malt_dir = 'data/malt'
malt_model = 'uni-dep-tb-ptbr'

# label of the dependency relation indicating negation 
negation_rel = 'neg'


# =============
# Tagger config
# =============
#

senna_path = r'D:\ferramentas\senna'
nlpnet_path_en = r'D:/desenvolvimento/nlpnet/data/dependency'

# ========================
# Lexical resources config
# ========================

stopwords_path = None

# pickle is faster to read than wordnet in nt or ppdb in txt
ownpt_path = 'data/own-pt.pickle'
ppdb_path = 'data/ppdb-xxl-phrasal.pickle'

lda_path = r'data/lda-100/lda.dat'
tfidf_path = r'data/lda-100/tfidf.dat'
tfidf_dict_path = r'data/lda-100/token-dict.dat'

unitex_dictionary_path = 'data/Delaf2015v04.dic'
