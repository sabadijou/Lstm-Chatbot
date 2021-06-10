import pysrt
import re
import os

# Load Subtitles #################################################################
sub_list = os.listdir(r'srt_files')
subs_text = ""

for sub in sub_list :
    try:
        sub_ = pysrt.open(r'srt_files/' + sub, encoding= 'utf-8')
        subs_text += sub_.text
        print(sub + " : is imported")
    except :
        print(sub + " : This subtitle is not suitable")

def remove_punctuation(doc) :
    my_punct = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
                '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
                '`', '{', '|', '}', '~', '»', '«', '“', '”', '؟', '،', '-', '♪']
    doc = re.sub("[" + re.escape("".join(my_punct)) + "]", ' ', str(doc))
    doc = re.sub(r'\s+', ' ' ,doc)
    return doc

def preprossing(documrct) :
    doc = re.sub(r'\d+', ' ', str(documrct))
    doc = re.sub(r'\s*[A-Za-z]+\b', ' ', str(doc))
    doc = re.sub(r'\s*[A-Za-z]+\b', ' ', str(doc))
    return doc

preprossed_doc = preprossing(subs_text).splitlines()
questions = []
answers = []
i = 5

while i < len(preprossed_doc) - 5 :
    if '؟' in preprossed_doc[i] :
        questions.append(remove_punctuation(preprossed_doc[i]).strip())
        answers.append(remove_punctuation(preprossed_doc[i + 1]).strip())
    i += 1

try:
    with open('questions.txt', 'w') as f:
        for item in questions:
            f.write("%s\n" % item)
    with open('answers.txt', 'w') as f:
        for item in answers:
            f.write("%s\n" % item)
except Exception as e :
    print(e)

