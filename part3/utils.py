import string
import numpy as np
from nltk import TweetTokenizer
tknzr = TweetTokenizer()

# ENDVALUE = 'theend'

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
#     sequence.append(word_library[ENDVALUE])
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
#         elif word == ENDVALUE:
#             return sonnet
        else:
            line.append(word)
    return sonnet
            
def vectorize_sonnets(sonnets):
    word_library = {'\n':0}#, ENDVALUE:1}
    feat_library = {0:'\n'}#, 1:ENDVALUE}
    num_features = 1#2
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
    
def get_syllable_dict(datafile = './data/Syllable_dictionary.txt'):
    syllable_dict = {} 
    with open(datafile) as f:
        for line in f:
            l = line.split(' ')
            word = l[0]
            word = word.replace("'", "")
            word = word.replace("-", "")        
            # Check if word is already in dictionary
            if not word in syllable_dict.keys():
                for i in range(len(l)):
                    if i == 0:
                        continue
                    try:
                        syllable_dict[word] = int(l[i][0])
                        break
                    except:
                        pass
    return syllable_dict

def count_syllables(line, syllable_dict=get_syllable_dict()):
    syllable_count = 0
    for word in line:
        punc = False
        for i in range(len(word)):
            if word[i] in string.punctuation:
                punc = True
        if not punc:
            syllable_count += syllable_dict[word]
    return syllable_count

def serenade_me_oh_sonneteer(model, feat_library):
    seq = np.empty((0,1),int)
    start_new_line = 0
    for i in range(14):
        syllable_count = 0
        if i in [4,8,12]:
            seq = np.vstack((seq, [[0]]))
        while syllable_count != 10:
            test_seq, test_states = model.sample(n_samples=20)
            try:
                end_line = np.asarray(test_seq[start_new_line:,0] == 0).nonzero()[0][0]
            except:
                continue
            test_seq = test_seq[:end_line+1]
            if test_seq[0].item == "\n": # explicitly call .item to avoid this warning https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
                continue
            test_line = sequence_to_sonnet(test_seq, feat_library)[0]
            syllable_count = count_syllables(test_line)
        seq = np.vstack((seq, test_seq[:end_line+1]))
    print_sonnet(sequence_to_sonnet(seq, feat_library))