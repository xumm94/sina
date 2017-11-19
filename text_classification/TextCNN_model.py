import tensorflow as tf
import numpy as np


class TextCNNConfig(object):

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = [2, 3, 4, 5]  # 卷积核尺寸
    #kernel_size = 3
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape = [None, self.config.seq_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, shape = [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

        self.cnn()


    def cnn(self):


        embedding = tf.get_variable(name = 'embedding', shape = [self.config.vocab_size, self.config.embedding_dim])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_outputs = []

        for _, filter_size in enumerate(self.config.kernel_size):
            with tf.variable_scope("convolution-pooling-%s" %filter_size):
                conv_layer = tf.layers.conv1d(inputs = embedding_input,
                                              filters= self.config.num_filters,
                                              kernel_size= filter_size,
                                              name = 'conv_layer-%s'%filter_size)
                max_pool = tf.reduce_max(input_tensor= conv_layer, reduction_indices= [1], name = 'max_pool-%s'%filter_size)
                pooled_outputs.append(max_pool)

        feature = tf.concat(pooled_outputs, axis = 1, name = 'feature')

        with tf.name_scope('score'):
            fc = tf.layers.dense(inputs = feature, units = self.config.hidden_dim, name = 'fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name = 'fc2')
            self.y_pre = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits= self.logits)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pre)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

