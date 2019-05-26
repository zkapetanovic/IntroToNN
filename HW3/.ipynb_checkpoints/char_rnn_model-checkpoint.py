import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class Model():
    def __init__(self, learning_rate, batch_size, num_layers, epochs, num_units, input_size, num_seq, rnn_type):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_seq = num_seq
        
        tf.reset_default_graph()

        self.X = tf.placeholder(tf.float32, [None, num_seq, input_size])
        self.Y = tf.placeholder(tf.float32, [None, None])

        #### LSTM ####
        lstm_layer = [tf.contrib.rnn.BasicRNNCell(num_units=num_units, activation=tf.nn.relu) for layer in range(num_layers)]

        #### Basic RNN ####
        rnn_layer = [tf.contrib.rnn.BasicLSTMCell(num_units=num_units, activation=tf.nn.relu) for layer in range(num_layers)]

        #### GRU ####
        gru_layer = [tf.contrib.rnn.GRUCell(num_units=num_units, activation=tf.nn.relu) for layer in range(num_layers)]

        layers = [lstm_layer, rnn_layer, gru_layer]

        multi_layer = tf.contrib.rnn.MultiRNNCell(layers[rnn_type])
        rnn_out, states = tf.nn.dynamic_rnn(multi_layer, self.X, dtype=tf.float32)
        stacked_outputs = tf.layers.dense(tf.reshape(rnn_out, [-1, num_units]), input_size)
        
        self.out = tf.reshape(stacked_outputs, [-1, num_seq, input_size])
        self.out = self.out[:,num_seq-1,:]
        
        self.loss = tf.reduce_mean(tf.square(self.out-self.Y)) #MSE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_op = optimizer.minimize(self.loss)
        

    def sample(self):
        return