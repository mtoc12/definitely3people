# hmm.py
# If you're running into trouble, execute `python tokenize.py` first to create the augmented syllable dictionary.
# The three tunable parameters are passed in at the bottom of this file

import re
import numpy as np
import os
from hmmlearn import hmm

def create_syllable_dictionary():
    syl_dict = {}
    code_dict = {}
    if os.path.exists("Dylan/syllable_dictionary_augmented.txt"):
        print("Found shakespeare + spenser syllable dictionary!")
        syl_dict_file = "Dylan/syllable_dictionary_augmented.txt"
    else:
        print("Found shakespeare-only syllable dictionary!")
        syl_dict_file = "Dylan/syllable_dictionary.txt"
    with open(syl_dict_file,"r") as syl:
        for index, line in enumerate(syl):
            word = line.split()[0].strip()
            syl_dict[word] = index+1
            code_dict[index+1] = word
    return syl_dict, code_dict, syl_dict_file

def read_in_sonnets(syl_dict_file):
    sonnets = []
    if syl_dict_file == "Dylan/syllable_dictionary_augmented.txt":
        sonnet_files = ["Dylan/shakespeare_tokenized.txt", "Dylan/spenser_tokenized.txt"]
    elif syl_dict_file == "Dylan/syllable_dictionary.txt":
        sonnet_files = ["Dylan/shakespeare_tokenized.txt"]
    else:
        raise ValueError
    for sonnet_file in sonnet_files:
        with open(sonnet_file,"r") as translation:
            for line in translation:
                if line == "\n":
                    try:
                        sonnet
                    except UnboundLocalError:
                        pass
                    else:
                        sonnets.append(sonnet)
                    sonnet = []
                else:
                    words = line.split(" ")
                    words = [int(word.strip()) for word in words]
                    sonnet.append(words)
            sonnets.append(sonnet)
    return sonnets

def read_in_syllable_counts():
    # get word-based syllable count dict
    syllable_counts = {}
    with open("Dylan/syllable_dictionary_augmented.txt","r") as syl_counts:
        for syl_count in syl_counts:
            line_split = syl_count.split(" ")
            if len(line_split)==2:
                try:
                    syls = int(line_split[1].strip("\n"))
                except ValueError:
                    print(line_split)
                    import pdb; pdb.set_trace()
                syllable_counts[line_split[0]] = [syls, syls]
            elif len(line_split)==3:
                #print(line_split)
                first_syls = int(line_split[1].strip("\nE"))
                second_syls = int(line_split[2].strip("\nE"))
                if first_syls < second_syls:
                    syllable_counts[line_split[0]] = [second_syls, first_syls]
                elif second_syls < first_syls:
                    syllable_counts[line_split[0]] = [first_syls, second_syls]
                else:
                    raise ValueError
            else:
                print(line_split)
                import pdb; pdb.set_trace()
                raise ValueError
    # change word-based dict to code-based one
    syl_dict, _, _ = create_syllable_dictionary()
    new_syl_dict = {}
    for syl in syllable_counts:
        new_syl_dict[ syl_dict[syl] ] = syllable_counts[syl]
    return new_syl_dict

def pad_line(syllable_counts, line):
    padded_line = []
    line_len = len(line)
    current_word = 0
    for word in line:
        current_word = current_word + 1
        if current_word < line_len:
            word_syl_count = syllable_counts[word][0]
        elif current_word == line_len:
            word_syl_count = syllable_counts[word][1]
        else:
            raise ValueError
        try:
            pads_needed = word_syl_count - 1
        except TypeError:
            import pdb; pdb.set_trace()
        # append word and pads to padded line
        padded_line.append(word)
        for pad_num in range(pads_needed):
            padded_line.append(0)
    return padded_line, len(padded_line)


def main(n_hmm_comps = 10, padding_type="end", learning_level = "sonnet"):
    np.random.seed(42)

    model = hmm.MultinomialHMM(n_components=n_hmm_comps)

    syl_dict, code_dict, syl_dict_file = create_syllable_dictionary()

    # get all the sonnets and make sure they all have 14 lines
    sonnets = read_in_sonnets(syl_dict_file)
    sonnet_number = 0
    author = "shakespeare"
    for sonnet in sonnets:
        sonnet_number = sonnet_number + 1
        if len(sonnet) != 14:
            print("Sonnet " + str(sonnet_number) + ", " + author)
            print("sonnet length: " + str(len(sonnet)))
            print(sonnet)
            raise ValueError("SonnetError")
        if sonnet_number == 152:
            sonnet_number = 0 # resetting when we finish with shakespeare and start counting spenser's sonnets
            author = "spenser"

    # pad lines: either add padding to end or inline to correspond to syllable number
    padding = padding_type
    if padding=="end":
        max_line_length = 10
        snp = np.array([[line + [0] * (max_line_length - len(line)) for line in sonnet] for sonnet in sonnets])
    elif padding=="inline":
        syllable_counts = read_in_syllable_counts()
        padded_lines = []
        line_lengths = []
        for sonnet in sonnets:
            for line in sonnet:
                padded_line, line_length = pad_line(syllable_counts, line)
                padded_lines.append(padded_line)
                line_lengths.append(line_length)
        snp = np.array([item for sublist in padded_lines for item in sublist])
    else:
        raise ValueError

    # determine level of input string: line or sonnet
    level = learning_level
    if level=="sonnet":
        #reshape at sonnet level
        if padding=="end":
            new_snp = snp.reshape(152*14*10,1)
            lengths = [140] * 152
        elif padding=="inline":
            new_snp = snp.reshape(snp.shape[0],1)
            # need to combine every 14 line lengths...
            sonnet_lengths = []
            for index, line_len in enumerate(line_lengths):
                if index % 14 == 0:
                    try:
                        sonnet_length
                    except:
                        pass
                    else:
                        sonnet_lengths.append(sonnet_length)
                    sonnet_length = 0
                else:
                    sonnet_length = sonnet_length + line_len
            sonnet_lengths.append(sonnet_length)
            lengths = sonnet_lengths
        else:
            raise ValueError
        model.fit(new_snp, lengths)
        snt = model.sample(n_samples=140)[0].tolist()
    elif level=="line":
        #reshape at line level
        if padding=="end":
            new_snp = snp.reshape(152*14*10,1)
            lengths = [10] * 14 * 152
        elif padding=="inline":
            new_snp = snp.reshape(snp.shape[0],1)
            lengths = line_lengths
        else:
            raise ValueError
        model.fit(new_snp, lengths)
        snt = []
        for line in range(14):
            snt.append(model.sample(n_samples=10)[0].tolist())
        snt = [item for sublist in snt for item in sublist]
    else:
        raise ValueError
    
    code_dict[0] = ""
    snt_trans = []
    for word in snt:
        snt_trans.append(code_dict[word[0]])
    snt_np = np.array(snt_trans)
    snt_np = snt_np.reshape(14,10)
    print(snt_np)
    
    # compute test sonnet's perplexity
    sonnet_perplexity = compute_perplexity(snt_trans)
    print("Perplexity of generated sonnet: " + str(sonnet_perplexity))
    

def compute_perplexity(sonnet):
    # Inspiration:
    # https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
    #
    # Just feed it a flat list of words and it'll give you their perplexity, relative to
    # the combined Shakespeare/Spenser corpus. The get preplexity relative to just Shakespeare,
    # just comment out the Spenser lines immediately below here.
    
    import collections, nltk, re
    from decimal import Decimal
    
    # clean up text for tokenization
    corpus_lines = []
    with open('Dylan/shakespeare_redacted.txt', 'r') as shakespeare_file:
        for shake_line in shakespeare_file:
            if re.match("     ", shake_line):
                pass
            else:
                corpus_lines.append(shake_line)
    corpus_lines.append("\n")
    with open('Dylan/spenser_redacted.txt', 'r') as spenser_file:
        for spense_line in spenser_file:
            if re.match("[LXVI]+", spense_line):
                pass
            else:
                corpus_lines.append(spense_line)
    corpus = " ".join(corpus_lines)

    # tokenize corpus text
    tokens = nltk.word_tokenize(corpus)

    # contruct unigram model
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N

    # compute perplexity
    sonnet = [ word for word in sonnet if word != "" ]
    testset = sonnet
    perplexity = Decimal(1)
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * Decimal(1/model[word])
    perplexity = pow(perplexity, Decimal(1/N))
    return perplexity

if __name__=='__main__':
    #import pdb; pdb.set_trace()
    main(7, "inline", "line")
