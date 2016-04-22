# -*- coding: utf-8 -*-

import tensorflow as tf

import numericautoencoder


class AdditionAutoEncoder(numericautoencoder.NumericAutoEncoder):
    """
    Subclass of the numeric autoencoder for modeling the addition operation.
    """

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
        embedded_add_sequence = self.generate_embeddings_list(self.sum)
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
