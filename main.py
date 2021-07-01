from tensorflow.keras.preprocessing.sequence import pad_sequences
from word_dict import dictionary, inverse_dict
from tensorflow.keras.models import load_model
from gensim.models import FastText
import numpy as np
import pickle
import re
import nltk

class chatbot():

    def __init__(self):
        self.enc_model = load_model(r'Model/encoder/enc.5h')
        self.dec_model = load_model(r'Model/decoder/decm.5h')
        self.fasttext = FastText.load(r'Model/model.bin')
        with open(r'Model/dense/dense.config', 'rb') as config_dictionary_file:
            self.dense = pickle.load(config_dictionary_file)

    def remove_punctuation(self, doc):
        my_punct = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
                    '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
                    '`', '{', '|', '}', '~', '»', '«', '“', '”', '؟', '،', '-', '♪']
        doc = re.sub("[" + re.escape("".join(my_punct)) + "]", ' ', str(doc))
        doc = re.sub(r'\s+', ' ', doc)
        return doc

    def response(self, question):
        my_dictionary = dictionary()
        question = [self.remove_punctuation(question)]
        word_puntuation_tokenizer = nltk.WordPunctTokenizer()
        word_toknized_corpus = [word_puntuation_tokenizer.tokenize(word) for word in question][0]
        question = []
        for word in word_toknized_corpus:
            try:
                question.append(my_dictionary[word])
            except:
                most_similar = self.fasttext.wv.similar_by_word(word, topn=1)
                question.append(my_dictionary[most_similar[0][0]])
        question = pad_sequences(sequences= [question], maxlen=20, dtype=int, padding='post',
                                truncating='post')
        stat = self.enc_model.predict(question)
        dec_s = np.zeros((1, 1))
        dec_s[0,0] = my_dictionary['<SOS>']
        inv_dict = inverse_dict()
        run = True
        ans = []
        while run :
            dec_outputs, h, c = self.dec_model.predict([dec_s] + stat)
            decoder_input = self.dense(dec_outputs)
            word_index = np.argmax(decoder_input[0, -1, :])
            inf_word = inv_dict[word_index]
            if inf_word == '<EOS>':
                run = False
            else:
                print(inf_word)
                stat = [h, c]
                dec_s = np.zeros((1, 1))
                dec_s[0, 0] = word_index
        return ans
if __name__ == '__main__':
    chat = chatbot()
    try:
        print("Press Ctrl+C to Exit")
        while True:
            user_question = input('Ask your Question :')
            ans = chat.response(user_question)
            print('Answer : {h}'.format(h = ans))
    except KeyboardInterrupt:
        pass