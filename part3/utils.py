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

    # convert words to lower case
    for i, sonnet in enumerate(sonnets):
        for j, line in enumerate(sonnet):
            for k, word in enumerate(line):
                sonnets[i][j][k] = word.lower()

    return sonnets

def sonnet_to_sequence(sonnet, word_library):
    sequence = []
    for line in sonnet:
        for word in line:
            sequence.append(word_library[word])
        sequence.append(word_library['\n'])
    sequence.append(word_library['\end'])
    return sequence

def sequence_to_sonnet(sequence, feat_library):
    sonnet = []
    line = []
    for item in sequence:
        if item.shape: # get value from array
            item = item[0]
        word = feat_library[item]
        if word == '\n':
            sonnet.append(line)
            line = []
        elif word == '\end':
            return sonnet
        else:
            line.append(word)
    return sonnet
            
def vectorize_sonnets(sonnets):
    word_library = {'\n':0, '\end':1}
    feat_library = {0:'\n', 1:'\end'}
    num_features = 2
    for sonnet in sonnets:
        for line in sonnet:
            for word in line:
                # Check if word is already in dictionary
                if not word in word_library.keys():
                    word_library[word] = num_features
                    feat_library[num_features] = word
                    num_features += 1

    sequences = []
    lengths = []
    for sonnet in sonnets:
        next_seq = sonnet_to_sequence(sonnet, word_library)
        sequences += next_seq
        lengths.append(len(next_seq))

    return sequences, lengths, word_library, feat_library, num_features

def print_sonnet(sonnet):
    string = ''
    for line in sonnet:
        for word in line:
            string += word
            string += ' '
        string += '\n'
    print(string)
