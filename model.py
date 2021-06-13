from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from word_dict import wc, dictionary, inverse_dict
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from gensim.models import FastText
import numpy as np
import nltk
import re


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
        self.enc_model = None
        self.dec_inp = None
        self.dec_lstm = None
        self.dec_model = None
        self.dec_embed = None
        self.dense = None
        self.words_number = 3040

    def preprocess_dataset(self, documnet, t = 0):
        try:
            word_puntuation_tokenizer = nltk.WordPunctTokenizer()
            word_toknized_corpus = [word_puntuation_tokenizer.tokenize(sent) for sent in documnet]
            i = 0
            my_dict = dictionary()
            final_vec = []
            if t == 0 :
                while i < len(word_toknized_corpus):
                    sent_vec = []
                    for word in word_toknized_corpus[i]:
                        sent_vec.append(my_dict[word.strip()])
                    final_vec.append(sent_vec)
                    i += 1
            if t == 1 :
                while i < len(word_toknized_corpus):
                    sent_vec = []
                    sent_vec.append(my_dict['<SOS>'])
                    for word in word_toknized_corpus[i]:
                        sent_vec.append(my_dict[word.strip()])
                    sent_vec.append(my_dict['<EOS>'])
                    final_vec.append(sent_vec)
                    i += 1
        except Exception as e:
            print(e)
            print("Vectorization procedure is stopped")

        return np.array(final_vec)

    def load_dataset(self):
        try:
            questions_lines = open(self.path + 'questions.txt', mode= 'r').readlines()
            answers_lines = open(self.path + 'answers.txt', mode= 'r').readlines()
        except Exception as e:
            print(e)
            return False
        self.questions = self.preprocess_dataset(questions_lines)
        self.answers = self.preprocess_dataset(answers_lines, t=1)
        self.questions = pad_sequences(sequences=self.questions, maxlen=20, dtype=int, padding='post',
                                           truncating='post')
        self.answers = pad_sequences(sequences=self.answers, maxlen=20, dtype=int, padding='post',
                                       truncating='post')
        self.final_answers = self.answers[:, range(1, 20)]
        self.final_answers = pad_sequences(sequences=self.final_answers, maxlen=20, dtype=int, padding='post',
                                       truncating='post')
        self.final_answers = to_categorical(self.final_answers, wc())
        return True

    def model(self):
        word_count = wc()
        print(type(word_count),word_count)
        enc_inp = Input(shape=(20, ))
        self.dec_inp = Input(shape=(20, ))
        # Embedding
        embed = Embedding(word_count+1,
                          output_dim=50,
                          input_length=20,
                          trainable=True)
        # Encoder
        enc_embed = embed(enc_inp)
        enc_lstm = LSTM(400, return_state=True, return_sequences=True)
        enc_op, h, c = enc_lstm(enc_embed)
        enc_states = [h, c]

        # Decoder
        self.dec_embed = embed(self.dec_inp)
        self.dec_lstm = LSTM(400, return_state=True, return_sequences=True)
        dec_op, _, _ = self.dec_lstm(enc_embed)
        dense_l = Dense(word_count, activation='softmax')
        self.dense = dense_l
        dense_op = dense_l(dec_op)
        enc_dec_model = Model([enc_inp, self.dec_inp], dense_op)
        enc_dec_model.compile(loss='categorical_crossentropy',
                              metrics=['acc'],
                              optimizer='adam')
        enc_dec_model.fit([self.questions, self.answers],
                          self.final_answers,
                          epochs=30)
        enc_dec_model.save(r'Model/lstm/edm.5h')
        self.enc_model = Model([enc_inp], enc_states)
    def inference(self):
        decoder_state_input_h = Input(shape=(400,))
        decoder_state_input_c = Input(shape=(400,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.dec_lstm(self.dec_embed, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        dec_model = Model([self.dec_inp] + decoder_state_inputs,
                          [decoder_outputs] + decoder_states)
        self.dec_model = dec_model

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


if __name__ == '__main__':
    x = chatbot_trainig()
    x.model()
    x.inference()
    x.response('الان چی گفتی')



