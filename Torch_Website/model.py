import torch
import torch.nn as nn
import unicodedata
import string

all_letters = string.ascii_letters + ".,;'-"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

lines = readLines('./data/names/English.txt')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(1 + input_size + hidden_size, hidden_size)
        self.i20 = nn.Linear(1 + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmaz(dmin=1)

    def forward(self, ):