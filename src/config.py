corenlp_path = r'D:\ferramentas\stanford-corenlp-full-2015-04-20'
senna_path = r'D:\ferramentas\senna'
nlpnet_path_en = r'D:/desenvolvimento/nlpnet/data/dependency'
nlpnet_path_pt = r'D:\ferramentas\Maltparser-Universal-Tree-Bank-PT-BR\nlpnet-data'
palavras_endpoint = 'http://143.107.183.175:12680/services/service_palavras_flat.php'

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
