# -*- coding: utf-8 -*-

from __future__ import division

import tensorflow as tf
import numpy as np

import numericautoencoder
from utils import Symbol


class MemorizerAutoEncoder(numericautoencoder.NumericAutoEncoder):
    """
    Subclass of the numeric autoencoder for memorizing a number.
    """

    def run(self, session, input_sequence, sequence_size):
        """
        Run the encoder/decoder for the given inputs

        :param session: a tensorflow session
        :param input_sequence: an input array with shape (max_time_steps, batch_size)
        :param sequence_size: the actual size of each sequence in the batch (the
            size of the contents before padding)
        :return: a numpy array with the same shape as input_sequence (the answer)
        """
        encoder_feeds = {self.first_term: input_sequence,
                         self.first_term_size: sequence_size}
        hidden_state = session.run(self.state_1st_term,
                                   feed_dict=encoder_feeds)

        return self.decoder_loop(session, hidden_state)

    def compute_l2_loss(self):
        """
        L2 norm of the softmax weights (outputs)
        """
        return self.l2_constant * tf.nn.l2_loss(self.softmax_weights)

    def generate_decoder_input(self):
        """
        Return the list of inputs to the decoder network. It prepends the GO symbol to
        the sequence given as input to the encoder.

        :return: a list of tensors of shape [batch_size, embedding_size]
        """
        embedded_go = self._generate_batch_embedded_go(self.embedded_1st_term[0])

        return [embedded_go] + self.embedded_1st_term

    def get_initial_hidden_state(self):
        """
        :return: the hidden state of the encoder after reading the first number
        """
        return self.state_1st_term

    def generate_decoder_labels(self):
        """
        Generate the labels to train the decoder. These are the input digits
        followed by the END symbol.

        :return: a list of tensors of shape [batch_size] with the labels (digits)
        """
        input_as_list = [tf.squeeze(time_step)
                         for time_step in tf.split(0, self.num_time_steps, self.first_term)]
        batch_end = numericautoencoder.generate_batch_end(input_as_list[0])

        return input_as_list + [batch_end]

    def get_accuracy(self, session, data, sizes, ignore_end=True):
        """
        Get the prediciton accuracy on the supplied data.

        :param session: current tensorflow session
        :param data: numpy array with shape (num_time_steps, batch_size)
        :param sizes: actual size of each sequence in data
        :param ignore_end: if True, ignore the END symbol
        """
        answer = self.run(session, data, sizes)

        # if the answer is longer than it should, truncate it
        if len(answer) > len(data):
            answer = answer[:len(data)]

        hits = answer == data
        total_items = answer.size

        if ignore_end:
            non_end = data != Symbol.END
            hits_non_end = hits[non_end]
            total_items = np.sum(non_end)

        acc = np.sum(hits_non_end) / total_items
        return acc
