import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class TRNNConfig(object):

    embedding_dim = 64
    seq_length = 600
    num_classes = 10
    vocab_size = 5000

    num_layers= 1
    hidden_dim = 128


    dropout_keep_prob = 0.8
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10


class TextRNN(object):
    '''双向RNN模型'''

    def __init__(self, config):

        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

        self.rnn()

    def rnn(self):

        embedding = tf.get_variable('embedding', shape = [self.config.vocab_size, self.config.embedding_dim])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        lstm_output = embedding_input

        with tf.name_scope('rnn'):
            for i in range(self.config.num_layers):
                with tf.variable_scope('%s'%i):
                    lstm_fw_cell = rnn.BasicLSTMCell(self.config.hidden_dim)
                    lstm_bw_cell = rnn.BasicLSTMCell(self.config.hidden_dim)

                    if self.keep_prob is not None:
                        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob= self.keep_prob)
                        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob= self.keep_prob)

                        lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, lstm_output, dtype= tf.float32)
                        lstm_output = tf.concat(lstm_output, axis = 2)

            out_put = lstm_output[:, -1, :]

        with tf.name_scope('scores'):
            fc = tf.layers.dense(out_put, units= self.config.hidden_dim, name = 'fc')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, units= self.config.num_classes, name = 'logits')
            self.y_pre_cls = tf.arg_max(self.logits, 1)

        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= self.logits, labels= self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuary'):
            correct_pre = tf.equal(tf.argmax(self.input_y, 1), self.y_pre_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))


