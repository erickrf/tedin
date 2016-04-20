# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np

# digits 0-9
encoder_vocab = 10

# digits 0-9 and END symbol
decoder_vocab = 11


class Symbol(object):
    """
    Placeholder class for values used in the RNNs.
    """
    END = 10
    GO = 11


class NumericLSTM(object):
    """
    Class that encapsulates the encoder-decoder architecture to
    memorize and perform arithmetics operations on arbitrary numbers.
    """

    def __init__(self, embedding_size, num_time_steps, embedding_abs_max=1.0, clip_value=1.25):
        """
        Initialize the encoder/decoder and creates Tensor objects

        :param embedding_size: the size of the digit and number embeddings.
            (this is the same as the hidden state of the encoder)
        :param num_time_steps: the number of time steps in the input
            sequences. This number is just the max; smaller sequences can be
            padded with the END symbol. Tensorflow requires that this number
            be know a priori.
        """
        self.num_time_steps = num_time_steps
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.l2_constant = tf.placeholder(tf.float32, name='l2_constant')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.first_term = tf.placeholder(tf.int32, [num_time_steps, None], 'first_term')
        self.first_term_size = tf.placeholder(tf.int32, [None], 'first_term_size')

        # embeddings are shared between encoder and decoder
        shape = [encoder_vocab, embedding_size]
        self.embeddings = tf.Variable(tf.random_uniform(shape, -embedding_abs_max,
                                                        embedding_abs_max),
                                      name='embeddings')

        num_lstm_units = embedding_size
        lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(embedding_size, embedding_size,
                                                 initializer=lstm_initializer)

        self.embedded_1st_term = self.generate_encoder_input(self.first_term)
        decoder_input = self.generate_decoder_input(self.embedded_1st_term)

        with tf.variable_scope('encoder') as self.encoder_scope:
            _, state_1st_term = tf.nn.rnn(self.lstm_cell, self.embedded_1st_term,
                                          sequence_length=self.first_term_size,
                                          dtype=tf.float32)

        with tf.variable_scope('decoder') as self.decoder_scope:
            raw_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_input, state_1st_term,
                                                       self.lstm_cell)

        with tf.variable_scope('output_softmax') as softmax_scope:
            # softmax to map decoder raw output to digits
            shape = [num_lstm_units, decoder_vocab]
            initializer = tf.truncated_normal_initializer(0.0, 0.1)
            self.softmax_weights = tf.get_variable('weights', shape, tf.float32, initializer)

            initializer = tf.zeros_initializer([decoder_vocab])
            self.softmax_bias = tf.get_variable('bias', initializer=initializer)

        self.output_logits = self.project_output(raw_outputs, False)
        self._create_training_tensors(clip_value)
        self._create_running_tensors()

    def run(self, session, input_sequence, sequence_size):
        """
        Run the encoder/decoder for the given inputs

        :param session: a tensorflow session
        :param input_sequence: an input array with shape (max_time_steps, batch_size)
        :param sequence_size: the actual size of each sequence in the batch (the
            size of the contents before padding)
        :return: a numpy array with the same shape as input_sequence (the answer)
        """
        answer = []
        batch_size = input_sequence.shape[1]
        current_digit = [Symbol.GO] * batch_size

        encoder_feeds = {self.first_term: input_sequence,
                         self.first_term_size: sequence_size}
        hidden_state = session.run(self.embedded_1st_term,
                                   feed_dict=encoder_feeds)

        # this array control which sequences have already been finished by the
        # decoder, i.e., for which ones it already produced the END symbol
        sequences_done = np.zeros(batch_size, dtype=np.bool)
        while True:
            decoder_feeds = {self.decoder_step_state: hidden_state,
                             self.digit_step: current_digit}

            fetches = session.run([self.next_digit, self.decoder_new_state],
                                  feed_dict=decoder_feeds)
            current_digit, hidden_state = fetches

            # use an "additive" or in order to avoid infinite loops
            sequences_done |= (current_digit == Symbol.END)

            if sequences_done.all():
                break

            answer.append(current_digit)

        return np.vstack(answer)

    def _create_training_tensors(self, clip_value):
        """
        Create member variables related to training.
        """
        decoder_labels = self.generate_decoder_labels()

        # label_weights is a mandatory parameter to weight the importance of each class
        # we set one for all
        label_weights = [tf.ones_like(decoder_labels[0], dtype=tf.float32)
                         for _ in decoder_labels]

        labeled_loss = tf.nn.seq2seq.sequence_loss(self.output_logits,
                                                   decoder_labels,
                                                   label_weights)
        l2_loss = self.l2_constant * tf.nn.l2_loss(self.softmax_weights)
        loss = labeled_loss + l2_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

    def _create_running_tensors(self):
        """
        Create the tensors necessary for running the encoder/decoder with
        new inputs.
        """
        # we use the same intermediate results of the training part
        self.digit_step = tf.placeholder(tf.int32, [None], 'digit_step')
        shape = [None, self.lstm_cell.state_size]
        self.decoder_step_state = tf.placeholder(tf.float32, shape, 'decoder_step_state')

        # embed the input digits
        decoder_step_input = [tf.nn.embedding_lookup(self.embeddings, self.digit_step)]

        with tf.variable_scope(self.decoder_scope) as exec_time_decoder:
            exec_time_decoder.reuse_variables()
            output_and_state = tf.nn.seq2seq.rnn_decoder(decoder_step_input,
                                                         self.decoder_step_state,
                                                         self.lstm_cell)
        decoder_step_output, self.decoder_new_state = output_and_state

        step_logits = tf.nn.xw_plus_b(decoder_step_output[0], self.softmax_weights, self.softmax_bias)
        self.next_digit = tf.argmax(step_logits, 1)

    def generate_encoder_input(self, sequence_indices):
        """
        Generate the embedded input to the encoder RNN

        :param sequence_indices: a tensor of shape [num_time_steps, batch_size].
        :return: a list of tensors of shape [batch_size, embedding_size]
        """
        embedded_sequence = tf.nn.embedding_lookup(self.embeddings, sequence_indices)
        return [tf.squeeze(time_step, [0])
                for time_step in tf.split(0, self.num_time_steps, embedded_sequence)]

    def generate_decoder_input(self, encoder_input):
        """
        Return the list of inputs to the decoder network. It just prepends the GO symbol
        to the sequence.

        :param encoder_input: the list of (embedded) inputs to the decoder, as returned
            by `self.generate_encoder_input`
        :return: a list of tensors of shape [batch_size, embedding_size]
        """
        # create a tensor of 1's with the appropriate size and then multiply it by GO embeddings
        ones = tf.ones_like(encoder_input[0])
        embedded_go = tf.nn.embedding_lookup(self.embeddings, Symbol.GO)
        batch_embedded_go = ones * embedded_go

        return [batch_embedded_go] + encoder_input

    def generate_decoder_labels(self):
        """
        Generate the labels to train the decoder. These are the input digits
        followed by the END symbol.

        :return: a list of tensors of shape [batch_size] with the labels (digits)
        """
        input_as_list = [tf.squeeze(time_step)
                         for time_step in tf.split(0, self.num_time_steps, self.first_term)]

        # the END symbol is just a label; doesn't need embeddings
        ones = tf.ones_like(input_as_list[0])
        batch_end = ones * Symbol.END

        return input_as_list + [batch_end]

    def project_output(self, raw_outputs, return_softmax=False):
        """
        Multiply the raw_outputs by a weight matrix, add a bias and return the
        softmax distribution or the logits.

        :param return_softmax: if True, return the softmaxes. If False, return
            the logits
        """
        output_logits = [tf.nn.xw_plus_b(time_step, self.softmax_weights,
                                         self.softmax_bias)
                         for time_step in raw_outputs]

        if not return_softmax:
            return output_logits

        return [tf.nn.softmax(time_step) for time_step in output_logits]

