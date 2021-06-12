from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
import numpy as np
import nltk


class chatbot_trainig() :

    def __init__(self):
        self.words_dictionary = dict()
        self.id = 0
        self.path = r'dataset/'
        self.questions = None
        self.answers = None
        self.final_answers = None
        self.fasttext = FastText.load(r'Model/model.bin')
        self.load_dataset()
    def preprocess_dataset(self, documnet):
        try:
            word_puntuation_tokenizer = nltk.WordPunctTokenizer()
            word_toknized_corpus = [word_puntuation_tokenizer.tokenize(sent) for sent in documnet]
            list = []
            for item in word_toknized_corpus :
                list.append(self.fasttext.wv[item])
            list = pad_sequences(sequences=list, maxlen=30, dtype=float, padding='post', truncating='post')
        except Exception as e:
            print(e)
            print("Vectorization procedure is stopped")
        return np.array(list)

    def load_dataset(self):
        try:
            questions_lines = open(self.path + 'questions.txt', mode= 'r').readlines()
            answers_lines = open(self.path + 'answers.txt', mode= 'r').readlines()
        except Exception as e:
            print(e)
            return False
        self.questions = self.preprocess_dataset(questions_lines)
        self.answers = self.preprocess_dataset(answers_lines)
        self.final_answers = self.answers[: ,range(3,30) ,: ]
        self.final_answers = pad_sequences(sequences=self.final_answers, maxlen=30, dtype=float, padding='post', truncating='post')
        return True

    def model(self):
        pass

if __name__ == '__main__':
    x = chatbot_trainig()