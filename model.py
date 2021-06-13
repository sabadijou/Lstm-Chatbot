from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models import FastText
import numpy as np
import nltk
from word_dict import wc, dictionary, inverse_dict

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
                          epochs=1)
        enc_dec_model.save(r'Model/lstm/edm.5h')
        self.enc_model = Model([enc_inp], enc_states)
    def inference(self):
        decoder_state_input_h = Input(shape=(400,))
        decoder_state_input_c = Input(shape=(400,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.dec_lstm(self.dec_inp, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        dec_model = Model([self.dec_inp] + decoder_state_inputs,
                          [decoder_outputs] + decoder_states)
        self.dec_model = dec_model
    def response(self, question):
        question = [question.strip()]
        question = self.preprocess_dataset(question)
        self.empty_target_seq = self.preprocess_dataset(['SOS'])
        """stat = self.enc_model(self.empty_target_seq)
        self.dec_outputs, self.h, self.c = self.dec_model.predict([self.empty_target_seq] + stat)
        i = np.zeros((30, 60))
        for vec in question[0] :
            i[0] = vec
            self.dec_outputs, self.h, self.c = self.dec_model.predict([self.empty_target_seq] + stat)
            decoder_concat_input = self.dense(self.dec_outputs)
            d = np.array(decoder_concat_input)
            print()
            #self.fasttext.wv.similar_by_vector(d, topn=1)"""

        e_o_s = self.preprocess_dataset(['EOS'])
        stat = self.enc_model(question)
        i = 0
        while i < 10 :
            dec_outputs, h, c = self.dec_model.predict([self.empty_target_seq] + stat)
            decoder_concat_input = self.dense(dec_outputs)
            d = np.array(decoder_concat_input)
            d = np.reshape(d, (30, 60))
            for vec in d:
                x = self.fasttext.wv.similar_by_vector(vec, topn=1)
                #if x != 0 :
                print(x)
            stat = [h, c]
            self.empty_target_seq = decoder_concat_input
            i += 1
        """stop_condition = False
        decoded_translation = ''
        i = 0
        while i < 30 :
            dec_outputs, h, c = self.dec_model.predict([self.empty_target_seq] + stat)
            decoder_concat_input = self.dense(dec_outputs)
            print(decoder_concat_input.shape)
            d = np.array(decoder_concat_input)
            d = np.reshape(d, (30, 60))
            if d.all == e_o_s.all:
                stop_condition = True
            else:
                for vec in d:
                    print(self.fasttext.wv.similar_by_vector(vec, topn=1))
                self.empty_target_seq = decoder_concat_input
                stat = [h, c]
            i += 1"""
if __name__ == '__main__':
    x = chatbot_trainig()
    x.model()
    #x.inference()
    #x.response('بریم چرنوبیل')



