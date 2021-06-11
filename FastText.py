import re
import time
import matplotlib.pyplot as plt
import nltk.stem
from gensim.models import FastText
import wikipedia as wiki
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
import arabic_reshaper
from bidi.algorithm import get_display

#nltk.download('punkt')
#nltk.download('wordnet')

# Load Data ############################################

print('Loading Data')
"""iran = wiki.page("iran").content
tehran = wiki.page("tehran").content"""

file = open(r'create_persian_dataset/answers.txt', mode= 'r')
iran = file.read()
file.close()
file = open(r'create_persian_dataset/questions.txt', mode= 'r')
tehran = file.read()

# tokenize #############################################

iran = sent_tokenize(iran)
tehran = sent_tokenize(tehran)
iran.extend(tehran)

# Preprocessing ########################################

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

# FastText ########################################

print("training....")
start_time = time.time()
fasttext_model = FastText(word_toknized_corpus,
                          vector_size= 60,
                          window = 40,
                          min_count= 5,
                          sample = 1e-2,
                          sg = 1,
                          epochs = 10)
finish_time = time.time()
print('Model Trained in {h} seconds'.format(h= finish_time - start_time))

#Simillarity Section ########################################

target_words = ['بیمارستان', 'ناموسا', 'دانشمند']
samantically_simillar_words = {words : [item[0] for item in fasttext_model.wv.most_similar([words], topn= 5)]
                               for words in target_words}
similar_words = sum([[k] + v for k, v in samantically_simillar_words.items()], [])

# Creating 2D Vector ###########################################

word_vectors = fasttext_model.wv[similar_words]
pca = PCA(n_components= 2)
points = pca.fit_transform(word_vectors)

# Plot ########################################################

plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1])
for word_names, x, y in zip(similar_words, points[:, 0], points[:, 1]) :
    plt.annotate(get_display(arabic_reshaper.reshape(word_names)), xy= (x + 0.06, y + 0.03))
plt.show()

