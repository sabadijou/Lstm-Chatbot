import re
import time

import matplotlib.pyplot as plt
import nltk.stem
from gensim.models import FastText
import wikipedia as wiki
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA

#nltk.download('punkt')
#nltk.download('wordnet')
iran = wiki.page("iran").content
tehran = wiki.page("tehran").content

iran = sent_tokenize(iran)
tehran = sent_tokenize(tehran)
iran.extend(tehran)

def preprocess(doc) :
    doc = re.sub(r'\W', ' ',str(doc))
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', str(doc))
    doc = re.sub(r'\s+', ' ', str(doc), flags=re.I)
    doc = re.sub(r'\d+', ' ', str(doc))
    doc = doc.lower()
    tokens = doc.split()
    tokens = [WordNetLemmatizer.lemmatize(self=FastText, word=word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 3]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

final_corpus = [preprocess(sentence) for sentence in iran if sentence.strip() != '']

word_puntuation_tokenizer = nltk.WordPunctTokenizer()
word_toknized_corpus = [word_puntuation_tokenizer.tokenize(sent) for sent in final_corpus]
print(time.thread_time())
fasttext_model = FastText(word_toknized_corpus,
                          vector_size= 60,
                          window = 40,
                          min_count= 5,
                          sample = 1e-2,
                          sg = 1,
                          epochs = 10)
print(time.thread_time())

words = ['pahlavi', 'revolution', 'ahmadinejad', 'abadijou', 'army']
similar_word = {"key" : "value"}
similar_word.clear()
for item in words :
    similar_word[item] = fasttext_model.wv.most_similar([item], topn=5)

#########################################
samantically_simillar_words = {words : [item[0] for item in fasttext_model.wv.most_similar([words], topn= 5)]
                               for words in ['shah', 'khomeni', 'hijab', 'bam']}

similar_words = sum([[k] + v for k, v in samantically_simillar_words.items()], [])

############################################

word_vectors = fasttext_model.wv[similar_words]

pca = PCA(n_components= 2)

points = pca.fit_transform(word_vectors)
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1])

for word_names, x, y in zip(similar_words, points[:, 0], points[:, 1]) :
    plt.annotate(word_names, xy= (x + 0.06, y + 0.03))
plt.show()

