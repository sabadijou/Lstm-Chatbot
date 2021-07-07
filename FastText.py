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
from gensim.test.utils import get_tmpfile
from tensorflow.keras.preprocessing.sequence import pad_sequences


#nltk.download('punkt')
#nltk.download('wordnet')

# Load Data ############################################

print('Loading Data')
"""answers = wiki.page("answers").content
questions = wiki.page("questions").content"""

file = open(r'dataset/answers.txt', mode= 'r')
answers = file.read()
file.close()
file = open(r'dataset/questions.txt', mode= 'r')
questions = file.read()

# tokenize #############################################

answers = sent_tokenize(answers)
questions = sent_tokenize(questions)
answers.extend(questions)

# Preprocessing ########################################

def preprocess(doc) :
    doc = re.sub(r'\^[a-zA-Z]\s+', ' ', str(doc))
    doc = re.sub(r'\s+', ' ', str(doc), flags=re.I)
    doc = re.sub(r'\d+', ' ', str(doc))
    tokens = doc.split()
    tokens = [WordNetLemmatizer.lemmatize(self=FastText, word=word) for word in tokens]
    tokens = [word for word in tokens if (len(word) > 3)]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

final_corpus = [preprocess(sentence) for sentence in answers if sentence.strip() != '']
word_puntuation_tokenizer = nltk.WordPunctTokenizer()
word_toknized_corpus = [word_puntuation_tokenizer.tokenize(sent) for sent in final_corpus]
###################################################
merged_list = []
merged_list = word_toknized_corpus[0] + word_toknized_corpus[1]
print('Number of words :', len(set(merged_list)))

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
fasttext_model.save(r"Model/model.bin") #save model
#Simillarity Section ########################################

target_words = ['شیطان', 'نیروگاه','دانشمند' , 'شهر' , 'آزمایش' , 'دلقك' , 'كليسا', 'رفیق', 'ستاره' , 'تاریخ']
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

