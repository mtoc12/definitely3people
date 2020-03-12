import string
import numpy as np
from nltk import TweetTokenizer
tknzr = TweetTokenizer()

def load_sonnets(datafile = './data/shakespeare.txt'):
    with open(datafile, 'r') as file:
        text = file.read()

    # split file into lines and words
    lines = [tknzr.tokenize(line) for line in text.split('\n') if line.split()]

    # Strip punctuation
    for i, line in enumerate(lines):
        for j, _ in enumerate(line):
            punc = "'"
            for punc in string.punctuation:
                lines[i][j] = lines[i][j].replace(punc, "")
            if len(lines[i][j]) == 0:
                del lines[i][j]

    # split up each sonnet, remove line denoting sonnet number
    sonnets = []
    for i, line in enumerate(lines):
        if line[0].isdigit():
            sonnet_number = int(line[0]) - 1
            sonnets.append([])
        else:
            sonnets[sonnet_number].append(line)