
# one of corenlp, palavras or malt
parser = 'corenlp'

corenlp_url = r'http://localhost'
corenlp_port = 9000
# path to the models inside the server
corenlp_depparse_path = r'pt-model/dep-parser'
corenlp_pos_path = r'pt-model/pos-tagger.dat'

senna_path = r'D:\ferramentas\senna'
nlpnet_path_en = r'D:/desenvolvimento/nlpnet/data/dependency'
palavras_endpoint = 'http://143.107.183.175:12680/services/service_palavras_flat.php'

malt_jar = r'data/malt/maltparser-1.8.1/maltparser-1.8.1.jar'
malt_dir = 'data/malt'
malt_model = 'uni-dep-tb-ptbr'

stopwords_path = 'data/stopwords.txt'

# Open Wordnet PT in NT format
ownpt_path = 'data/own-pt.nt'

lda_path = r'data/lda-100/lda.dat'
tfidf_path = r'data/lda-100/tfidf.dat'
tfidf_dict_path = r'data/lda-100/token-dict.dat'

# URL dentro da rede NILC
#palavras_endpoint = 'http://10.11.14.126/services/service_palavras_flat.php'

import sklearn
import sklearn.linear_model as linear
classifier_class = sklearn.svm.SVC #linear.LogisticRegression
regressor_class = linear.LinearRegression
