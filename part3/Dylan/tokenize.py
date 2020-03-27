# tokenize.py
# Take in shakespeare and spenser files (after cleaning to remove any sonnets that are not 14 lines),
# tokenize them using a word-to-number lookup dict, and write them out again in
# shakespeare_translated.txt and spenser_translated.txt

import re
import os

class Tokenizer:
    def __init__(self):
        pass

    def create_syllable_dictionary(self):
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

    def write_out_augmented_syllable_dict(self, new_syls):
        with open("Dylan/syllable_dictionary_augmented.txt","w") as outfile:
            # copy original syllable dict entries
            with open("Dylan/syllable_dictionary.txt","r") as orig:
                for orig_line in orig:
                    outfile.write(orig_line)
            # add new entries
            for syl_tuple in new_syls:
                syl_string = syl_tuple[0] + " " + syl_tuple[1]
                outfile.write(syl_string + "\n")

    def tokenize(self, annotate_spenser_syllables=False):
        new_syls = []
        syl_dict, code_dict, syl_dict_file = self.create_syllable_dictionary()
        # define source sonnets according to syllable dictionary state
        if syl_dict_file == "Dylan/syllable_dictionary.txt":
            file_pairs = [("Dylan/shakespeare_redacted.txt", "Dylan/shakespeare_tokenized.txt")]
        elif syl_dict_file == "Dylan/syllable_dictionary_augmented.txt":
            file_pairs = [("Dylan/shakespeare_redacted.txt", "Dylan/shakespeare_tokenized.txt"), ("Dylan/spenser_redacted.txt", "Dylan/spenser_tokenized.txt")]
        else:
            raise ValueError
        # tokenize source sonnets
        for file_pair in file_pairs:
            original_file = file_pair[0]
            tokenized_file = file_pair[1]
            with open(tokenized_file, "w") as tokend:
                with open(original_file, "r") as orig:
                    for orig_line in orig:
                        if re.match("\n", orig_line):
                            pass
                        elif original_file == "Dylan/shakespeare_redacted.txt" and re.match("    ", orig_line):
                            tokend.write("\n")
                        elif original_file == "Dylan/spenser_redacted.txt" and re.match("[LXVI]+", orig_line):
                            tokend.write("\n")
                        else:
                            words = [word.strip(':,.;?()!').lower() for word in orig_line.split()]
                            if annotate_spenser_syllables:
                                # add all new words to syl_dict
                                for word in words:
                                    if word not in syl_dict:
                                        syl_dict[word] = max(syl_dict.values()) + 1
                                        syls = input(word + " ")
                                        new_syls.append( (word, syls) )
                            try:
                                tokenization = [syl_dict[word] for word in words]
                            except KeyError:
                                #print(words)
                                # sometimes, a word begins or ends with a single quote which isn't part of the word
                                # let's try stripping those, then strip everything else, and try again
                                words = [word.strip('\'') for word in words]
                                words = [word.strip(':,.;?()!').lower() for word in words]
                                try:
                                    tokenization = [syl_dict[word] for word in words]
                                except KeyError:
                                    print(words)
                                    # ok, we have an actual problem
                                    import pdb; pdb.set_trace()
                                    raise KeyError
                            tokenization = [str(token) for token in tokenization]
                            token_string = " ".join(tokenization)
                            token_string = token_string + "\n"
                            tokend.write(str(token_string))
        if annotate_spenser_syllables:
            self.write_out_augmented_syllable_dict(new_syls)

if __name__=='__main__':
    Tokenizer().tokenize()
