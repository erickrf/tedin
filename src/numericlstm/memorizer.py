# -*- coding: utf-8 -*-

import tensorflow as tf

import numericautoencoder


class MemorizerAutoEncoder(numericautoencoder.NumericAutoEncoder):
    """
    Subclass of the numeric autoencoder for memorizing a number.
    """

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