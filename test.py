import os

with open(r'create_persian_dataset/answers.txt') as f :
    lines = f.readlines()
    f.close()
new_lines = []
for line in lines :
    new_line = line.strip()
    new_lines.append(new_line)

with open(r'dataset/answers.txt', 'w') as f:
    for item in new_lines:
        f.write("%s\n" % item)

"""        predicted_word = enc_dec_model.predict(
            [self.questions[1730].reshape(1, 30, 60), self.answers[1730].reshape(1, 30, 60)])
        predicted_word = predicted_word.reshape(30, 60)

        for vec in predicted_word:
            print(self.fasttext.wv.similar_by_vector(vec))
            print('\n')"""


