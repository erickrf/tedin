# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import os
import tensorflow as tf

from .tree_edit_network import TreeEditDistanceNetwork
from ..datastructures import Dataset


class PairRanker(object):
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
        self.tedin2 = TreeEditDistanceNetwork(
            params, self.session, self.tedin1.embeddings, reuse_weights=True,
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

    def _run(self, extra_feeds, fetches, batch_positive, batch_negative):
        """
        Run the model with the given fetches for a batch.

        :param extra_feeds: dictionary with training hyperparams, if needed
        :param fetches: op or ops to fetch
        :param batch_positive: Dataset object with "positive" sentence pairs
        :param batch_negative: Dataset object with "negative" sentence pairs
        :return: the computed fetches
        """
        operation_feed1 = self.tedin1.create_operation_feeds(batch_positive)
        operation_feed2 = self.tedin2.create_operation_feeds(batch_negative)

        # update modifies the dictionary in-place, but this is not a problem
        # it will be modified again for the next batch
        extra_feeds.update(operation_feed1)
        extra_feeds.update(operation_feed2)

        return self.session.run(fetches, extra_feeds)

    def run_validation(self, data, batch_size=None):
        """
        Run the model on validation data and return the loss.

        :param data: tuple (positive, negative) of Datasets
        :param batch_size: None or integer. If None, the whole dataset will be
            evaluated at once, which may not fit memory.
        :return: tuple (accuracy, loss) as python floats
        """
        positive_data, negative_data = data
        min_size = min(len(positive_data), len(negative_data))
        if batch_size is None:
            batch_size = min_size

        positive_data.reset_batch_counter()
        negative_data.reset_batch_counter()

        accumulated_loss = 0
        num_batches = int(min_size / batch_size)

        for _ in range(num_batches):
            pos_batch = positive_data.next_batch(batch_size, wrap=False)
            neg_batch = negative_data.next_batch(batch_size, wrap=False)
            loss = self._run({}, self.loss, pos_batch, neg_batch)

            # multiply by len(batch) because the last batch may have a different
            # length
            accumulated_loss += loss * len(pos_batch)

        loss = accumulated_loss / min_size

        return loss

    def train(self, train_data, valid_data, params, model_dir, report_interval):
        """
        Train the model

        :param train_data: tuple of Datasets (positive, negative)
        :param valid_data: tuple of Datasets (positive, negative)
        :param params: TedinParameters
        :param model_dir: path to the model dir
        :param report_interval:
        :return:
        """
        best_loss = 1e10
        accumulated_train_loss = 0
        loss_denominator = params.batch_size * report_interval

        path = os.path.join(model_dir, self.filename)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)

        feeds = {self.learning_rate: params.learning_rate,
                 self.dropout_keep: params.dropout}
        positive_data, negative_data = train_data

        self.logger.info('Starting training')
        for step in range(params.num_steps):
            # TODO: avoid some repeated code with TEDIN
            pos_batch = positive_data.next_batch(params.batch_size, wrap=True,
                                                 shuffle=True)
            neg_batch = negative_data.next_batch(params.batch_size, wrap=True,
                                                 shuffle=True)

            fetches = [self.loss, self.train_op]
            loss, _ = self._run(feeds, fetches, pos_batch, neg_batch)
            accumulated_train_loss += loss

            if step % report_interval == 0:
                train_loss = accumulated_train_loss / loss_denominator
                accumulated_train_loss = 0
                msg = '{}/{} positive/negative epochs\t{} steps\t' \
                      'Train loss: {:.5}\tValid loss: {:.5}'
                valid_loss = self.run_validation(valid_data)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    saver.save(self.session, path, step)
                    msg += ' (saved model)'

                self.logger.info(msg.format(
                    positive_data.epoch, negative_data.epoch, step,
                    train_loss, valid_loss))

    def initialize(self, embeddings):
        """
        Initialize all trainable variables

        :param embeddings: numpy 2d array
        """
        self.session.run(tf.global_variables_initializer(),
                         {self.tedin1.embedding_ph: embeddings})
