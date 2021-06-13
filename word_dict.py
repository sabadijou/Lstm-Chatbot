import re
import nltk.stem
from gensim.models import FastText
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

class dictionary_class() :

    def __init__(self):
        self.word_dict = dict()
        self.answers = None
        self.questions = None
        self.word_toknized_corpus = None
        self.word_count = None
        self.merged_list = None

    def load_data(self):
        file = open(r'dataset/answers.txt', mode= 'r')
        self.answers = file.read()
        file.close()
        file = open(r'dataset/questions.txt', mode= 'r')
        self.questions = file.read()

    def preprocess(self, doc) :
        doc = re.sub(r'\^[a-zA-Z]\s+', ' ', str(doc))
        doc = re.sub(r'\s+', ' ', str(doc), flags=re.I)
        doc = re.sub(r'\d+', ' ', str(doc))
        tokens = doc.split()
        tokens = [WordNetLemmatizer.lemmatize(self=FastText, word=word) for word in tokens]
        #tokens = [word for word in tokens if (len(word) > 2)]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def toknize(self):
        a = self.answers
        q = self.questions
        answers = sent_tokenize(a)
        questions = sent_tokenize(q)
        answers.extend(questions)
        final_corpus = [self.preprocess(sentence) for sentence in answers if sentence.strip() != '']
        word_puntuation_tokenizer = nltk.WordPunctTokenizer()
        self.word_toknized_corpus = [word_puntuation_tokenizer.tokenize(sent) for sent in final_corpus]
        self.merged_list = self.word_toknized_corpus[0] + self.word_toknized_corpus[1]
        self.merged_list = set(self.merged_list)
        self.word_count = len(self.merged_list) + 2

    def create_dict(self):
        i = 0
        for word in self.merged_list :
            self.word_dict[word.strip()] = i
            i += 1
        self.word_dict['<SOS>'] = i
        self.word_dict['<EOS>'] = i + 1
    def wc(self):
        return self.word_count

    def dictionary(self):
        return self.word_dict


x = dictionary_class()
x.load_data()
x.toknize()
x.create_dict()
def wc():
    return x.word_count
def dictionary():
    return x.word_dict
def inverse_dict():
    inv = {v: k for k, v in x.word_dict.items()}
    return inv
