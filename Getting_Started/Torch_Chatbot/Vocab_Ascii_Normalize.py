
with open('formatted_movie_lines.txt', 'rb') as f:
    lines = f.readlines()

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token:'SOS', EOS_token: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    #removing words below a certain count threshold
    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('Keep words: {} / {} = {:.4f}'.format(len(keep_words),
                                                   len(self.word2index),
                                                   len(keep_words) / len(self.word2index)))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    import re
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    return s
