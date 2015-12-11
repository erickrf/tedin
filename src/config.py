corenlp_path = r'D:\ferramentas\stanford-corenlp-full-2015-04-20'
senna_path = r'D:\ferramentas\senna'
nlpnet_path_en = r'D:/desenvolvimento/nlpnet/data/dependency'
nlpnet_path_pt = r'D:/desenvolvimento/nlpnet/data/pos-pt'
palavras_endpoint = 'http://143.107.183.175:12680/services/service_palavras_flat.php'

lda_path = r'data\lda-100\lda.dat'
tfidf_path = r'data\lda-100\tfidf.dat'
tfidf_dict_path = r'data\lda-100\token-dict.dat'

# URL dentro da rede NILC
#palavras_endpoint = 'http://10.11.14.126/services/service_palavras_flat.php'

import sklearn.linear_model as linear
classifier_class = linear.LogisticRegression
regressor_class = linear.LinearRegression
