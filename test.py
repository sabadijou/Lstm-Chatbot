import os

with open(r'create_persian_dataset/answers.txt') as f :
    lines = f.readlines()
    f.close()
new_lines = []
for line in lines :
    new_line = '<SOS>' + line.strip() + '<EOS>'
    new_lines.append(new_line)

with open(r'dataset/answers.txt', 'w') as f:
    for item in new_lines:
        f.write("%s\n" % item)


