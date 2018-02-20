# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from .tree_edit_network import TreeEditDistanceNetwork
from .base import Trainable


class PairRanker(Trainable):
    """
    Class that learns parameters for tree edit distance comparison by training
    on pairs of unrelated and related trees.
    """
    filename = 'tedin'

    def __init__(self, params, session=None):
        """
        Initialize a TreeComparisonNetwork.

        :param params: TedinParameters for the contained TEDIN
        """
        self.params = params

        # training params
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
        self.dropout_keep = tf.placeholder_with_default(1., None,
                                                        'dropout_keep')

        # The network is trained with pairs positive and negative
        # there is a TreeEditDistanceNetwork for each
        self.tedin1 = TreeEditDistanceNetwork(params, create_optimizer=False,
                                              session=session)
        self.session = self.tedin1.session
        self.tedin2 = TreeEditDistanceNetwork(
            params, self.session, self.tedin1.embeddings, reuse_weights=True,
            create_optimizer=False)
        self.logger = self.tedin1.logger

        # tedin1 holds the "positive" pairs; tedin2 holds the "negative" ones
        # both distance variables have shape (batch,)
        distances1 = self.tedin1.transformation_cost
        distances2 = self.tedin2.transformation_cost

        # the loss is given as max(0, 1 - (distance2 - distance1))
        # i.e. negative pairs should have a higher distance than positive ones
        diff = distances2 - distances1
        loss = tf.maximum(0., 1 - diff)

        tcl1 = self.tedin1.transformation_cost_loss
        tcl2 = self.tedin2.transformation_cost_loss
        self.loss = tf.reduce_mean(loss) + tcl1 + tcl2

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def _create_base_training_feeds(self, params):
        feeds = {self.learning_rate: params.learning_rate,
                 self.dropout_keep: params.dropout,
                 self.tedin1.cost_regularizer: params.cost_regularizer,
                 self.tedin2.cost_regularizer: params.cost_regularizer}
        return feeds

    def _get_next_batch(self, data, batch_size, training):
        pos_data, neg_data = data
        pos_batch = pos_data.next_batch(batch_size, wrap=training,
                                        shuffle=training)
        neg_batch = neg_data.next_batch(batch_size, wrap=training,
                                        shuffle=training)

        if not training and len(pos_batch) != len(neg_batch):
            # in validation, make sure that both batches have the same size
            # this is not necessary in training because the dataset is wrapped
            # around
            min_len = min(len(pos_batch), len(neg_batch))
            if len(pos_batch) > len(neg_batch):
                pos_batch = pos_batch[:min_len]
            else:
                neg_batch = neg_batch[:min_len]

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
        self._loss_denominator = report_interval

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

        # both datasets in the batch must have the same size at this point
        self._accumulated_validation_loss += loss * len(pos_data)

    def _get_validation_metrics(self, data):
        return self._accumulated_validation_loss / self._get_data_size(data)

    def _get_data_size(self, data):
        pos_data, neg_data = data
        return min(len(pos_data), len(neg_data))

    def _validation_report(self, values, saver, step, train_data, model_dir):
        valid_loss = values
        train_loss = self._accumulated_training_loss / self._loss_denominator
        self._accumulated_training_loss = 0

        msg = '{}/{} positive/negative epochs\t{} steps\t' \
              'Train loss: {:.5}\tValid loss: {:.5}'

        if valid_loss < self._best_loss:
            path = self.tedin1.get_base_file_name(model_dir)
            saver.save(self.session, path)
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

    @classmethod
    def load(cls, path, embeddings, session=None):
        """
        Load a saved model

        :param path: directory with saved model
        :param embeddings: numpy embedding matrix
        :param session: existing tensorflow session to be reuse, or None to
            create a new one
        :return: PairRanker instance
        """
        params = cls.load_params(path)

        if session is None:
            session = tf.Session()

        ranker = PairRanker(params, session)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(session, cls.get_base_file_name(path))
        session.run(tf.variables_initializer([ranker.tedin1.embeddings]),
                    {ranker.tedin1.embedding_ph: embeddings})
        return ranker

    def evaluate(self, data, batch_size=256):
        """
        Evaluate the model on the given dataset

        :param data: Dataset
        :param batch_size: batch size, important if the data don't fit all in
            memory
        :return: average loss
        """
        # certify that both positive and negative data have the same number of
        # items; if not, truncate to the smaller
        pos_data, neg_data = data
        if len(pos_data) != len(neg_data):
            min_len = min(len(pos_data), len(neg_data))
            pos_data = pos_data[:min_len]
            neg_data = neg_data[:min_len]

        data = (pos_data, neg_data)
        acc_loss = [0]

        def accumulate(values, batch, acc_loss=acc_loss):
            # workaround to persist values accross function call without
            # setting an attribute
            batch_loss = values[0]
            acc_loss[0] += batch_loss * self._get_data_size(batch)

        self._main_loop(data, accumulate, [self.loss], batch_size)
        loss = acc_loss[0] / len(pos_data)

        return loss
