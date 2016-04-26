# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np

from utils import Symbol

# digits 0-9, END and GO symbols
# maybe END and GO could be merged somehow to save one embedding entry
# but it seems more trouble than its worth... an additional embedding
# entry that is never used doesn't need to be trained anyway
input_vocab = 12

# digits 0-9 and END symbol
decoder_vocab = 11


class NumericAutoEncoder(object):
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
        self.second_term = tf.placeholder(tf.int32, [num_time_steps, None], 'second_term')
        self.second_term_size = tf.placeholder(tf.int32, [None], 'second_term_size')
        self.sum = tf.placeholder(tf.int32, [num_time_steps+1, None], 'sum')

        # embeddings are shared between encoder and decoder
        shape = [input_vocab, embedding_size]
        self.embeddings = tf.Variable(tf.random_uniform(shape, -embedding_abs_max,
                                                        embedding_abs_max),
                                      name='embeddings')

        num_lstm_units = embedding_size
        lstm_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(embedding_size, embedding_size,
                                                 initializer=lstm_initializer)

        with tf.variable_scope('addition_layer') as self.addition_scope:
            # the addition layer has as input a concatenation of two encoded arrays
            # its output is the input for the decoder LSTM
            shape = [2 * self.lstm_cell.state_size, self.lstm_cell.state_size]
            initializer = tf.truncated_normal_initializer(0.0, 0.1)
            self.addition_weights = tf.get_variable('weights', shape, initializer=initializer)

            initializer = tf.zeros_initializer([self.lstm_cell.state_size])
            self.addition_bias = tf.get_variable('bias', initializer=initializer)

        with tf.variable_scope('output_softmax') as self.softmax_scope:
            # softmax to map decoder raw output to digits
            shape = [num_lstm_units, decoder_vocab]
            initializer = tf.truncated_normal_initializer(0.0, 0.1)
            self.softmax_weights = tf.get_variable('weights', shape, tf.float32, initializer)

            initializer = tf.zeros_initializer([decoder_vocab])
            self.softmax_bias = tf.get_variable('bias', initializer=initializer)

        self.embedded_1st_term = self.generate_embeddings_list(self.first_term)
        self.embedded_2nd_term = self.generate_embeddings_list(self.second_term)

        with tf.variable_scope('encoder') as self.encoder_scope:
            _, self.state_1st_term = tf.nn.rnn(self.lstm_cell, self.embedded_1st_term,
                                               sequence_length=self.first_term_size,
                                               dtype=tf.float32)
            self.encoder_scope.reuse_variables()
            _, self.state_2nd_term = tf.nn.rnn(self.lstm_cell, self.embedded_2nd_term,
                                               sequence_length=self.second_term_size,
                                               dtype=tf.float32)

        two_terms = tf.concat(1, [self.state_1st_term, self.state_2nd_term])
        self.addition_output = tf.nn.xw_plus_b(two_terms, self.addition_weights,
                                               self.addition_bias)

        initial_hidden_state = self.get_initial_hidden_state()
        decoder_input = self.generate_decoder_input()

        with tf.variable_scope('decoder') as self.decoder_scope:
            raw_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_input,
                                                       initial_hidden_state,
                                                       self.lstm_cell)

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
        hidden_state = session.run(self.state_1st_term,
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

    def get_initial_hidden_state(self):
        """
        Subclasses should implement
        """
        raise NotImplementedError

    def generate_decoder_input(self):
        """
        Subclasses should implement
        """
        raise NotImplementedError

    def generate_decoder_labels(self):
        """
        Subclasses should implement
        """
        raise NotImplementedError

    def compute_l2_loss(self):
        """
        Subclasses should implement
        """
        raise NotImplementedError

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
        self.loss = labeled_loss + self.compute_l2_loss()

        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

    def _create_running_tensors(self):
        """
        Create the tensors necessary for running the encoder/decoder with
        new inputs.
        """
        # we use the same intermediate results of the training part

        # digit_step stores each digit predicted by the decoder
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

    def generate_embeddings_list(self, sequence_indices, num_time_steps=None):
        """
        Generate a list with the embeddings corresponding to the sequence_indices at
        each time step.

        :param sequence_indices: a tensor of shape [num_time_steps, batch_size].
        :return: a list of tensors of shape [batch_size, embedding_size]
        """
        num_time_steps = num_time_steps or self.num_time_steps
        embedded_sequence = tf.nn.embedding_lookup(self.embeddings, sequence_indices)
        return [tf.squeeze(time_step, [0])
                for time_step in tf.split(0, num_time_steps, embedded_sequence)]

    def _generate_batch_embedded_go(self, like):
        """
        Generate a tensor with the embeddings for the GO symbol repeated as many times
        as the size of the `like` parameter.

        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor with shape as `like`
        """
        # create a tensor of 1's with the appropriate size and then multiply it by GO embeddings
        ones = tf.ones_like(like)
        embedded_go = tf.nn.embedding_lookup(self.embeddings, Symbol.GO)
        return ones * embedded_go

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


def generate_batch_end(like):
    """
    Generate a tensor with the (non-embedded) END symbol repeated as many times
    as the size of the `like` parameter.

    :param like: a tensor whose shape the returned embeddings should match
    :return: a tensor with shape as `like`
    """
    ones = tf.ones_like(like)
    return ones * Symbol.END

