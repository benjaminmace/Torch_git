import torch
import csv
import codecs

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

line_fields = ['lineID', 'characterID', 'movieID', 'character', 'text']
lines = {}

with open('movie_lines.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(' +++$+++ ')
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineID']] = lineObj

conv_fields = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']
conversations = []

with open('movie_conversations.txt', 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split(' +++$+++ ')
        convObj = {}
        for i, field in enumerate(conv_fields):
            convObj[field] = values[i]
        lineIds = eval(convObj['utteranceIDs'])
        convObj['lines'] = []
        for lineId in lineIds:
            convObj['lines'].append(lines[lineId])
        conversations.append(convObj)

qa_pairs = []
for conversation in conversations:
    for i in range(len(conversation['lines']) - 1):
        inputLine = conversation['lines'][i]['text'].strip()
        targetLine = conversation['lines'][i+1]['text'].strip()
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])


deli = '\t'

delimiter = str(codecs.decode(deli, 'unicode_escape'))

print('Writing newly formatted file...')
with open('formatted_movie_lines.txt', 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)

print('Done writing!')