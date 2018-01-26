# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import tensorflow as tf

from tree_edit_network import TreeEditDistanceNetwork
from trainable import Trainable


class TreeComparisonNetwork(Trainable):
    """
    Class that learns parameters for tree edit distance comparison by training
    on pairs of unrelated and related trees.
    """
    def __init__(self, params):
        """
        Initialize a TreeComparisonNetwork.

        :param params: an instance of TedinParameters
        """
        super(TreeComparisonNetwork, self).__init__(has_accuracy=False)

        # training params
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
        self.dropout_keep = tf.placeholder(tf.float32, None, 'dropout_keep')

        # The network is trained with pairs positive and negative
        # there is a TreeEditDistanceNetwork for each
        self.tedin1 = TreeEditDistanceNetwork(self.session,
                                              params.num_hidden_units)
        self.tedin2 = TreeEditDistanceNetwork(self.session,
                                              params.num_hidden_units,
                                              reuse_weights=True)

        # tedin1 holds the "positive" pairs; tedin2 holds the "negative" ones
        # both distance variables have shape (batch,)
        distances1 = self.tedin1.tree_distances
        distances2 = self.tedin2.tree_distances

        # the loss is given as max(0, 1 - (distance1 - distance2))
        diff = distances1 - distances2
        loss = tf.maximum(0, 1 - diff)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def run_batch(self, fetches, batch_positive, batch_negative):
        """
        Run the model with the given fetches for a batch.

        :param fetches: list of tensorflow fetches
        :param batch_positive: Dataset object with "positive" sentence pairs
        :param batch_negative: Dataset object with "negative" sentence pairs
        :return: the computed fetches
        """
        feeds = self.tedin1.create_batch_feed(batch_positive)
        feeds2 = self.tedin2.create_batch_feed(batch_negative)
        feeds.update(feeds2)

        return self.session.run(fetches, feeds)
