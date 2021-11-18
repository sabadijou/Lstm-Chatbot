import pandas as pd
import re

df = pd.read_csv(r'question_answer_2.csv', header=None)

df.to_csv("digi_questions.txt", columns=[1], header=False, index=False)
df.to_csv("digi_answers.txt", columns=[2], header=False, index=False)

with open('digi_questions.txt', 'r') as f:
    q = f.readlines()
    f.close()
with open('digi_answers.txt', 'r') as f:
    a = f.readlines()
    f.close()

def remove_punctuation(doc) :
    my_punct = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
                '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
                '`', '{', '|', '}', '~', '»', '«', '“', '”', '؟', '،', '-', '♪', '‫' ]
    doc = re.sub("[" + re.escape("".join(my_punct)) + "]", ' ', str(doc))
    doc = re.sub(r'\s+', ' ' ,doc)
    return doc

def preprossing(documrct) :
    doc = re.sub(r'[a-zA-Z0-9]+', ' ', str(documrct))
    doc = re.sub(r'\d+', ' ', str(doc))
    doc = re.sub(r'\s*[A-Za-z]+\b', ' ', str(doc))
    doc = re.sub(r'\s*[A-Za-z]+\b', ' ', str(doc))
    return doc

length = len(q)

i = 0

while i < length :
    q[i] = preprossing(q[i])
    q[i] = remove_punctuation(q[i])
    a[i] = preprossing(a[i])
    a[i] = remove_punctuation(a[i])
    i += 1
i = 0
while i < length :
    if q[i] == " " or a[i] == " ":
        q[i] = 'سلام'
        a[i] = 'سلام'
    i +=1
try:
    with open('digi_questions.txt', 'w') as f:
        for item in q:
            f.write("%s\n" % item)
    with open('digi_answers.txt', 'w') as f:
        for item in a:
            f.write("%s\n" % item)
except Exception as e :
    print(e)
