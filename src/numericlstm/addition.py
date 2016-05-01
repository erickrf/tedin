# -*- coding: utf-8 -*-

import tensorflow as tf

import numericautoencoder


class AdditionAutoEncoder(numericautoencoder.NumericAutoEncoder):
    """
    Subclass of the numeric autoencoder for modeling the addition operation.
    """

    def run(self, session, first_term, second_term, first_size, second_size):
        """
        Run the autoencoder for the given data.

        :param session: tensorflow session
        :param first_term: numpy array with shape (sequence_size, batch_size)
        :param second_term: same as first_term
        :param first_size: numpy array with the actual sequence sizes
        :param second_size: same as first_size
        :return: a numpy array with the results, digit by digit, with shape
            (time_steps, batch_size)
        """
        sum_feeds = {self.first_term: first_term,
                     self.second_term: second_term,
                     self.first_term_size: first_size,
                     self.second_term_size: second_size}
        hidden_state = session.run(self.addition_output, feed_dict=sum_feeds)

        return self.decoder_loop(session, hidden_state)

    def compute_l2_loss(self):
        """
        L2 norm of the softmax weights (outputs) plus the L2 of the addition weights
        """
        base = tf.nn.l2_loss(self.softmax_weights) + tf.nn.l2_loss(self.addition_weights)
        return self.l2_constant * base

    def generate_decoder_input(self):
        """
        Return the list of inputs to the decoder network. It embeds the expected output sentence
        and prepends the GO symbol to it.

        :return: a list of tensors of shape [batch_size, embedding_size]
        """
        embedded_add_sequence = self.generate_embeddings_list(self.sum, self.num_time_steps + 1)
        embedded_go = self._generate_batch_embedded_go(embedded_add_sequence[0])

        return [embedded_go] + embedded_add_sequence

    def get_initial_hidden_state(self):
        """
        :return: the output of the addition layer
        """
        return self.addition_output

    def generate_decoder_labels(self):
        """
        Generate the labels to train the decoder. These are the digits of the sum result
        followed by the END symbol.

        :return: a list of tensors of shape [batch_size] with the labels (digits)
        """
        input_as_list = [tf.squeeze(time_step)
                         for time_step in tf.split(0, self.num_time_steps+1, self.sum)]
        batch_end = numericautoencoder.generate_batch_end(input_as_list[0])

        return input_as_list + [batch_end]
