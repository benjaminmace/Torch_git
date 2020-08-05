from Vocab_Ascii_Normalize import Vocabulary, normalizeString

voc = Vocabulary("cornell movie-dialogs corpus")

lines = open('formatted_movie_lines.txt', encoding='utf-8').read().strip().split('\n')

pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]

MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

pairs = [pair for pair in pairs if len(pair) > 1]

pairs = filterPairs(pairs)

for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
print('Counted words:', voc.num_words)

for pair in pairs[:10]:
    print(pair)