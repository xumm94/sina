import tensorflow as tf
import numpy as np
import copy

class TextRCNNConfig(object):

    embedding_dim = 64
    seq_length = 600
    num_classes = 10
    num_filters = 256
    kernel_size = [2, 3, 4, 5]
    #kernel_size = 3
    vocab_size = 5000

    hidden_dim = 128

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10


class TextRCNN(object):

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name= 'keep_prob')

        self.activation = tf.nn.tanh
        self.embedding_input = None

        self.weight_init()
        self.RCNN()

    def weight_init(self):

        with tf.name_scope('weight_init'):
            self.Embedding = tf.get_variable(name = 'Embedding',
                                        shape = [self.config.vocab_size, self.config.embedding_dim])

            self.left_side_first_word = tf.get_variable(name = 'left_side_first_word',
                                                   shape = [self.config.batch_size, self.config.embedding_dim])
            self.right_side_last_word = tf.get_variable(name =  'right_side_last_word',
                                                   shape = [self.config.batch_size, self.config.embedding_dim])

            self.W_l = tf.get_variable(name = 'W_l', shape = [self.config.embedding_dim, self.config.embedding_dim])
            self.W_R = tf.get_variable(name = 'W_R', shape = [self.config.embedding_dim, self.config.embedding_dim])
            self.W_sl = tf.get_variable(name = 'W_sl', shape = [self.config.embedding_dim, self.config.embedding_dim])
            self.W_sR = tf.get_variable(name = 'W_sR', shape = [self.config.embedding_dim, self.config.embedding_dim])

    def get_context_left(self,context_left,embedding_previous):

        #left_c = tf.matmul(context_left, self.W_l)
        #left_e = tf.matmul(embedding_previous, self.W_sl)
        left_c = tf.layers.dense(context_left, units=self.config.embedding_dim, use_bias=False)
        left_e = tf.layers.dense(embedding_previous, units=self.config.embedding_dim, use_bias=False)
        left_h = left_c + left_e
        context_left = self.activation(left_h)
        return context_left

    def get_context_right(self, context_right, embedding_afterward):

        #right_c = tf.matmul(context_right, self.W_R)
        right_c = tf.layers.dense(context_right, units=self.config.embedding_dim, use_bias= False)
        #right_e = tf.matmul(embedding_afterward, self.W_sR)
        right_e =  tf.layers.dense(embedding_afterward, units=self.config.embedding_dim, use_bias= False)
        self.shape = tf.shape(right_e)
        right_h = right_c + right_e
        context_right = self.activation(right_h)
        return context_right


    def conv_layer_with_recurrent_structure(self):

        embedding_words = tf.split(self.embedding_input, self.config.seq_length, axis = 1)
        #embedding_words: a list of  [None, 1, self.config.embedding_dim]
        embedding_words_squeeze = [tf.squeeze(x, axis = 1) for x in embedding_words]

        embedding_words_shape = tf.shape(embedding_words)

        left_word_pre = self.left_side_first_word
        left_context_pre = tf.zeros([embedding_words_shape[1], self.config.embedding_dim])

        context_left_list = []

        for _, embedding_word in enumerate(embedding_words_squeeze):
            context_left = self.get_context_left(left_context_pre, left_word_pre)
            context_left_list.append(context_left)
            left_word_pre = embedding_word
            left_context_pre = context_left

        embedding_words_squeeze_reverse = copy.copy(embedding_words_squeeze)
        embedding_words_squeeze_reverse.reverse()

        right_word_after = self.right_side_last_word
        right_context_after = tf.zeros([embedding_words_shape[1], self.config.embedding_dim])
        context_right_list = []

        for _, embedding_word in enumerate(embedding_words_squeeze_reverse):
            context_right = self.get_context_right(right_context_after, right_word_after)
            context_right_list.append(context_right)
            right_context_after = context_right
            right_word_after = embedding_word

        context_right_list.reverse()

        output_list = []
        for index, embedding_word in enumerate(embedding_words_squeeze_reverse):
            reprensent = tf.concat([context_left_list[index], embedding_word, context_right_list[index]], axis = 1)
            output_list.append(reprensent)

        output = tf.stack(output_list, axis = 1)

        return output


    def RCNN(self):

        self.embedding_input = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        output_conv = self.conv_layer_with_recurrent_structure()
        output_pool = tf.reduce_max(output_conv, axis = 1)

        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(output_pool, keep_prob= self.keep_prob)

        with tf.name_scope('output'):
            self.logits = tf.layers.dense(h_drop, units= self.config.num_classes, use_bias= True)

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels= self.input_y, logits= self.logits)
            self.loss = tf.reduce_mean(loss)
            self.optim = tf.train.AdamOptimizer(learning_rate= self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('acc'):
            pre = tf.argmax(self.logits, 1)
            self.y_pred_cls = pre
            pre = tf.cast(pre, dtype= tf.float32)
            label = tf.cast(tf.argmax(self.input_y, 1), dtype= tf.float32)
            acc = tf.cast(tf.equal(pre, label), tf.float32)
            self.acc = tf.reduce_mean(acc)





