import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import io
from tqdm import tnrange, trange
"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
class TextLoader():
    def __init__(self, filename=None, batch_size=None, seq_len=None):
        
        if filename is not None:
            self.text = self.load_file(filename)
        
        self.chars = self.get_characters(self.text)
        self.char_to_int = self.char_2_int(self.chars)
        self.num_chars = len(self.text)
        self.num_vocab = len(self.chars)
        self.X, self.Y = self.gen_data_set(self.text, seq_len, self.num_chars, self.char_to_int, self.chars)
        
        self.X_train, self.Y_train = self.process_data_set(self.X, self.Y, seq_len, self.num_vocab)
    def load_file(self, filename):
        text = io.open(filename, encoding='utf-8').read().lower()
        return text
    
    def get_characters(self, text):
        chars = sorted(list(set(text)))
        return chars
    
    def char_2_int(slef, chars):
        char_to_int = dict((c,i) for i, c in enumerate(chars))
        return char_to_int
    
    def gen_data_set(self, text, seq_len, num_chars, char_to_int, char):

        X, Y = [], []

        for i in trange(0, num_chars-seq_len, 1):
            in_ = text[i: i + seq_len]
            out_ = text[i + seq_len]
            X.append([char_to_int[c] for c in in_])
            Y.append(char_to_int[out_])

        print('Number of Training Examples:', len(X))

        return X, Y
    
    def process_data_set(self, X, Y, seq_len, num_vocab):
        X_data = np.reshape(X, (len(X), seq_len,1))
        X_data = X_data/float(num_vocab)
        
        #Y_data = np_utils.to_categorical(Y)
        
        return X_data, np.reshape(Y, (len(Y), -1))
        
