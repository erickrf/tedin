# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import tensorflow as tf

from .tree_edit_network import TreeEditDistanceNetwork
from .base import Trainable


class PairRanker(Trainable):
    """
    Class that learns parameters for tree edit distance comparison by training
    on pairs of unrelated and related trees.
    """
    filename = 'pair-ranker'

    def __init__(self, params):
        """
        Initialize a TreeComparisonNetwork.

        :param params: TedinParameters for the contained TEDIN
        """
        # training params
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
        self.dropout_keep = tf.placeholder_with_default(1., None,
                                                        'dropout_keep')

        # The network is trained with pairs positive and negative
        # there is a TreeEditDistanceNetwork for each
        self.tedin1 = TreeEditDistanceNetwork(params, create_optimizer=False)
        self.session = self.tedin1.session
        embedding_vars = [self.tedin1.embeddings, self.tedin1.label_embeddings]
        self.tedin2 = TreeEditDistanceNetwork(
            params, self.session, embedding_vars, reuse_weights=True,
            create_optimizer=False)
        self.logger = self.tedin1.logger

        # tedin1 holds the "positive" pairs; tedin2 holds the "negative" ones
        # both distance variables have shape (batch,)
        distances1 = self.tedin1.transformation_cost
        distances2 = self.tedin2.transformation_cost

        # the loss is given as max(0, 1 - (distance1 - distance2))
        diff = distances1 - distances2
        loss = tf.maximum(0., 1 - diff)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def _create_base_training_feeds(self, params):
        feeds = {self.learning_rate: params.learning_rate,
                 self.dropout_keep: params.dropout}
        return feeds

    def _get_next_batch(self, data, batch_size, training):
        pos_data, neg_data = data
        pos_batch = pos_data.next_batch(batch_size, wrap=training,
                                        shuffle=training)
        neg_batch = neg_data.next_batch(batch_size, wrap=training,
                                        shuffle=training)
        return pos_batch, neg_batch

    def _create_data_feeds(self, data):
        pos_data, neg_data = data
        feeds1 = self.tedin1._create_data_feeds(pos_data)
        feeds2 = self.tedin2._create_data_feeds(neg_data)
        feeds1.update(feeds2)

        return feeds1

    def _init_train_stats(self, params, report_interval):
        self._best_loss = 1e10
        self._accumulated_training_loss = 0
        self._loss_denominator = params.batch_size * report_interval

    def _init_validation(self, data):
        self._accumulated_validation_loss = 0
        data[0].reset_batch_counter()
        data[1].reset_batch_counter()

    def _update_training_stats(self, values, data):
        loss, _ = values
        self._accumulated_training_loss += loss

    def _update_validation_stats(self, values, data):
        loss = values[0]
        pos_data, neg_data = data
        self._accumulated_validation_loss += loss * len(pos_data)

    def _get_validation_metrics(self, data):
        return self._accumulated_validation_loss / self._get_data_size(data)

    def _get_data_size(self, data):
        pos_data, neg_data = data
        return min(len(pos_data), len(neg_data))

    def _validation_report(self, values, saver, step, train_data):
        valid_loss = values
        train_loss = self._accumulated_training_loss / self._loss_denominator
        self._accumulated_training_loss = 0

        msg = '{}/{} positive/negative epochs\t{} steps\t' \
              'Train loss: {:.5}\tValid loss: {:.5}'

        if valid_loss < self._best_loss:
            saver.save(self.session, self.path, step)
            self._best_loss = valid_loss
            msg += ' (saved model)'

        pos_epoch = train_data[0].epoch
        neg_epoch = train_data[1].epoch
        self.logger.info(msg.format(pos_epoch, neg_epoch, step, train_loss,
                                    valid_loss))

    def initialize(self, embeddings):
        """
        Initialize all trainable variables

        :param embeddings: numpy 2d array
        """
        self.session.run(tf.global_variables_initializer(),
                         {self.tedin1.embedding_ph: embeddings})
