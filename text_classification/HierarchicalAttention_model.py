import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib


class HierarchicalAttention_model(object):
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_len, num_sentences,
                 vocab_size, embed_size, hidden_size,
                 is_training, need_sentence_level_attention_encoder_flag = True,
                 initializer = tf.random_normal_initializer(stddev= 0.1),
                 clip_gradients = 5.0):

        '''init all hyperparameter here'''
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.clip_gradients = clip_gradients

        # add place holder

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_len], name = "input_x")
        self.sequence_len = int(self.sequence_len / self.num_sentences)

        self.input_y = tf.placeholder(tf.int32, [None], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name = "dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable= False, name = "Global_step")
        self.epoch_step = tf.Variable(0, trainable = False, name = "Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  #[None, label_size]

        self.predictions = tf.argmax(self.logits, axis = 1, name = "Predictions")

        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_x)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "Accuracy")

        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()


    def attention_word_level(self, hidden_state):
        '''
        :param hidden_state:list, len:sentence_length, element = [batch_size * num_sentence, hidden_size * 2]
        :return:sentence_representation: shape [bathc * num_sentence, hidden_size * 2]
        '''

        hidden_state_ = tf.stack(hidden_state, axis = 1)
        #shape [batch_size * num_sentence, sentence_length, hidden_size * 2]
        hidden_state_2 = tf.reshape(hidden_state_, shape = [-1, self.hidden_size * 2])
        #shape [bathc_size * num_sentence * sentence_length, hidden_size * 2]

        hidden_representation = tf.matmul(hidden_state_2, self.W_w_attention_word) + self.W_b_attention_word
        hidden_representation = tf.nn.tanh(hidden_representation) #shape = [bathc_size * num_sentence * sentence_length, hidden_size * 2]


        # attention process
        # 1) get logits for each word in sentence
        attention_logits = tf.matmul(hidden_representation, self.context_vector_word) #[batch_size * num_sentence * sentence_length]
        attention_logits = tf.reshape(attention_logits, [-1, self.sequence_len], name = "attention_logits") #shape = [batch_size * num_sentence , sentence_length]
        attention_logits_max = tf.reduce_max(attention_logits, axis = 1, keep_dims = True)

        # 2) get possibility distribution for each word in the sentence.

        p_attention = tf.nn.softmax(
            attention_logits - attention_logits_max
        )   #[batch_size * num_sentence, sentence_length]
        # 3) get weighted hidden state by attention vector
        p_attention = tf.expand_dims(p_attention, axis = 2) #shape = [batch_size * num_sentence, sentence_length, 1]

        sentence_representation = tf.multiply(hidden_state_, p_attention) #[batch_size * num_sentence, sentence_length, hidden_size * 2]
        sentence_representation = tf.reduce_sum(sentence_representation, axis = 1)

        return sentence_representation #shape [batch_size * num_sentence, hidden_size * 2]


    def attention_sentence_level(self, hidden_state):
        '''

        :param hidden_state:a list, length : num_sentences, elementary : [None, hidden_size * 4]

        :return:document representation [None, hidden_size * 4]
        '''

        hidden_state_ = tf.stack(hidden_state, axis = 1) #shape = [batch, num_sentence, hidden_size * 4]
        hidden_state_1 = tf.reshape(hidden_state_, [-1, self.hidden_size * 4]) #shape = [bathc_size * num_sentence, hidden_size * 4]

        hidden_representation = tf.matmul(hidden_state_1, self.W_w_attention_sentence) + self.W_b_attention_sentence  #shape = [batch_size * num_sentences, hidden_size * 2]
        hidden_representation = tf.nn.tanh(hidden_representation)


        #attention process
        #1) get attention logits
        attention_logits = tf.matmul(hidden_representation, self.context_vector_sentence) #shape = [batch_size * num_sentence]
        attention_logits = tf.reshape(attention_logits, [-1, self.num_sentences], name = "attention_logits")
        attention_logits_max = tf.reduce_max(attention_logits, axis = 1, keep_dims = True)

        # 2) get possibility distribution for each sentence in the document.
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max, dim = 1) #shape = [batch_size , num_sentence]

        # 3) get weighted hidden state by attention vector
        p_attention = tf.expand_dims(p_attention, axis = 2) #shape = [batch_size , num_sentence, 1]

        document_representation = tf.multiply(hidden_state_, p_attention)
        document_representation = tf.reduce_mean(document_representation, axis = 1) #shape = [batch_size, hidden_size * 4]

        return document_representation


    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        '''
        single step of gru for word level
        :param Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        '''

        #update gate
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_word) + tf.matmul(h_t_minus_1, self.U_z_word) + self.b_z_word) #shape = [batch_size*num_sentences, hidden_size]
        #reser gate
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_word) + tf.matmul(h_t_minus_1, self.U_r_word) + self.b_r_word) #shape = [batch_size*num_sentences, hidden_size]
        #candiate state h_t
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_word)
                                  + tf.multiply( r_t, tf.matmul(h_t_minus_1, self.U_h_word))
                                  + self.b_h_word) #shape = [batch_size*num_sentences, hidden_size]
        #new state
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate

        return h_t

    def gru_single_step_sentence_level(self, Xt, h_t_minus_1):
        '''

        :param Xt: [batch_size, embed_size]
        :param h_t_minus_1: [batch_size, embed_size]
        :return:
        '''

        #update gate
        z_t = tf.sigmoid(tf.matmul(Xt, self.W_z_sentence) + tf.matmul(h_t_minus_1, self.U_z_sentence) + self.b_z_sentence)
        #reset gate
        r_t = tf.sigmoid(tf.matmul(Xt, self.W_r_sentence) + tf.matmul(h_t_minus_1, self.U_r_sentence) + self.b_r_sentence)
        #candiate state h_t
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_sentence)
                                  + tf.multiply(r_t, tf.matmul(h_t_minus_1, self.U_h_sentence))
                                  + self.b_h_sentence)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t


    def gru_forward_word_level(self, embeded_words):
        '''

        :param embeded_words: [batch_size * num_sentences, sequence_len, embedded_size]
        :return:forword hidden state list : len = sequence_len, elementary : [batch_size * num_sentence, hidden_size]
        '''

        embedded_words_splitted = tf.split(embeded_words, self.sequence_len, axis = 1)
        #It is a list, len : self.sequence_len, elementary : [batch_size * num_sentence, 1, embedded_size]
        embeded_words_squeeze =  [tf.squeeze(x, axis = 1) for x in embedded_words_splitted]
        #It is a list. list len :sequence_len, elementary : [batch_size * num_sentence, embedded_size]

        h_t = tf.ones([self.batch_size * self.num_sentences, self.hidden_size])

        h_t_forward_list = []
        for time_step, Xt in enumerate(embeded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_forward_list.append(h_t)

        return h_t_forward_list


    def gru_backward_word_level(self, embeded_words):
        '''

        :param embeded_words: [batch_size * num_sentences, sequence_len, embedded_size]
        :return: backword hidden state list:len  sequence_len,, elementary :[batch_size * num_sentence,  hidden_size]
        '''

        embedded_words_splitted = tf.split(embeded_words, self.sequence_len, axis = 1)
        #It is a list, len : self.sequence_len, elementary : [batch_size * num_sentence, 1, embedded_size]
        embeded_words_squeeze =  [tf.squeeze(x, axis = 1) for x in embedded_words_splitted]
        #It is a list. list len :sequence_len, elementary : [batch_size * num_sentence, embedded_size]
        embeded_words_squeeze.reverse()

        h_t = tf.ones([self.batch_size * self.num_sentences, self.hidden_size])
        h_t_backward_list = []

        for time_step, Xt in enumerate(embeded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)

        h_t_backward_list.reverse()

        return h_t_backward_list

    def gru_forward_sentence_level(self, embeded_words):
        '''

        :param embeded_words: [batch_size , num_sentence, embed_size]
        :return: forward hidden state [batch_size, num_sentence, hidden_size]
        '''

        embeded_words_splitted = tf.split(embeded_words, self.num_sentences, axis = 1)
        #It is a list, len: num_sentence, elementary[batch_size, 1, embed_size]
        embeded_words_squeezed = [tf.squeeze(x, axis = 1) for x in embeded_words_splitted]
        # It is a list, len: num_sentence, elementary[batch_size, embed_size]

        h_t = tf.ones(self.batch_size, self.hidden_size)
        h_t_forward_list = []

        for time_step, Xt in enumerate(embeded_words_squeezed):
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_forward_list.append(h_t)

        return h_t_forward_list

    def gru_bacward_sentence_level(self, embeded_words):
        '''

        :param embeded_words: [batch_size , num_sentence, embed_size]
        :return: backward hidden state [batch_size, num_sentence, hidden_size]
        '''

        embeded_words_splitted = tf.split(embeded_words, self.num_sentences, axis=1)
        # It is a list, len: num_sentence, elementary[batch_size, 1, embed_size]
        embeded_words_squeezed = [tf.squeeze(x, axis=1) for x in embeded_words_splitted]
        # It is a list, len: num_sentence, elementary[batch_size, embed_size]

        embeded_words_squeezed.reverse()

        h_t = tf.ones(self.batch_size, self.hidden_size)
        h_t_backward_list = []

        for time_step, Xt in enumerate(embeded_words_squeezed):
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_backward_list.append(h_t)

        h_t_backward_list.reverse()
        return h_t_backward_list

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss


    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op


    def instantiate_weights(self):
        '''
        define all weights here
        :return:
        '''

        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape = [self.vocab_size, self.embed_size], initializer = self.initializer)
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape = [self.embed_size * 4, self.num_classes], initializer = self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape = [self.num_classes], initializer = self.initializer)

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_word_level"):
            #update gate
            self.W_z_word = tf.get_variable("W_z_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.U_z_word = tf.get_variable("U_z_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.b_z_word = tf.get_variable("b_z_word", shape = [self.hidden_size], initializer = self.initializer)

            #reset gate
            self.W_r_word = tf.get_variable("W_r_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.U_r_word = tf.get_variable("U_r_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.b_r_word = tf.get_variable("b_r_word", shape = [self.hidden_size], initializer = self.initializer)

            #candiate h
            self.W_h_word = tf.get_variable("W_h_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.U_h_word = tf.get_variable("U_h_word", shape = [self.embed_size, self.hidden_size], initializer = self.initializer)
            self.b_h_word = tf.get_variable("b_h_word", shape = [self.hidden_size], initializer = self.initializer)

        with tf.name_scope("gru_weights_sentence_level"):
            #update gate
            self.W_z_sentence = tf.get_variable("W_z_sentence", shape = [self.hidden_size * 2, self.hidden_size * 2], initializer = self.initializer)
            self.U_z_sentence = tf.get_variable("U_z_sentence", shape = [self.hidden_size * 2, self.hidden_size * 2], initializer = self.initializer)
            self.b_z_sentence = tf.get_variable("b_z_sentence", shape = [self.hidden_size * 2], initializer = self.initializer)

            #reset gate
            self.W_r_sentence = tf.get_variable("W_r_sentence", shape = [self.hidden_size * 2 , self.hidden_size * 2], initializer = self.initializer)
            self.U_r_sentence = tf.get_variable("U_r_sentence", shape = [self.hidden_size * 2, self.hidden_size * 2], initializer = self.initializer)
            self.b_r_sentence = tf.get_variable("b_r_sentence", shape = [self.hidden_size * 2], initializer = self.initializer)

            #candiate h
            self.W_h_sentence = tf.get_variable("W_h_word", shape = [self.hidden_size * 2, self.hidden_size * 2], initializer = self.initializer)
            self.U_h_sentence = tf.get_variable("U_h_word", shape = [self.hidden_size * 2, self.hidden_size * 2], initializer = self.initializer)
            self.b_h_sentence = tf.get_variable("b_h_word", shape = [self.hidden_size * 2], initializer = self.initializer)

        with tf.name_scope("attention"):

            #word level
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word",shape=[self.hidden_size * 2],initializer=self.initializer)
            self.context_vector_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2], initializer=self.initializer)

            #sentence level
            self.W_w_attention_sentence = tf.get_variable("W_w_attention_sentence",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable("W_b_attention_sentence",shape=[self.hidden_size * 2],initializer=self.initializer)
            self.context_vector_sentence = tf.get_variable("what_is_the_informative_sentence", shape=[self.hidden_size * 2], initializer=self.initializer)

    def inference(self):
        '''
        main computation graph here
        1.word encoding
        2.word attention
        3.sentence encoding
        4.sentence attention
        5.liner classifier
        '''

        #1 word attention
        #1.1 Embedding of word
        input_x = tf.split(value = self.input_x, num_or_size_splits = self.num_sentences, axis = 1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x = tf.stack(input_x, axis = 1) #shape = [batch_size, num_sentence, sequence_length]
        self.embeded_words = tf.nn.embedding_lookup(self.Embedding, input_x, name = "embeded_words") #shape = [batch_size, num_sentence, sequence_length, embed_size]
        embedded_words_reshaped = tf.reshape(self.embeded_words, shape = [-1, self.sequence_len, self.embed_size])

        #1.2 forward gru
        hidden_state_forward_list = self.gru_forward_word_level(embedded_words_reshaped) #list, len:sequence_length, elementary [batch_size * num_sentence, hidden_size]
        #1.3 backword gru
        hidden_state_backward_list = self.gru_backward_word_level(embedded_words_reshaped) #list, len:sequence_length, elementary [batch_size * num_sentence, hidden_size]
        #1.4 concat forward list and backward list, list, len = sequence_length, elementary  [batch_size * num_sentence, hidden_size * 2]
        self.hidden_state_word = [tf.concat([h_forward, h_backward],axis = 1) for h_forward, h_backward in zip(hidden_state_forward_list, hidden_state_backward_list)]

        #2 word attention
        #for each sentence
        sentence_representation = self.attention_word_level(self.hidden_state_word)
        sentence_representation = tf.reshape(sentence_representation, [-1, self.num_sentences, self.hidden_size * 2])

        #3 sentence encoding
        #3.1 forward gru
        hidden_state_forward_list = self.gru_forward_sentence_level(sentence_representation)
        #3.2 backward_gru
        hidden_state_backward_list = self.gru_bacward_sentence_level(sentence_representation)
        #3.3 concat forwardlist and backward list
        self.hidden_state_sentence = [tf.concat([h_forward, h_backward], axis = 1) for h_forward, h_backward in zip(hidden_state_forward_list, hidden_state_backward_list)]

        #4 sentence attention
        document_representation = self.attention_sentence_level(self.hidden_state_sentence)

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(document_representation, keep_prob = self.dropout_keep_prob)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope('output'):
            logits = tf.matmul(self.h_drop,self.W_projection) + self.b_projection

        return logits













