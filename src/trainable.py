# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import abc
import logging

import tensorflow as tf


def get_logger(name='logger'):
    """
    Setup and return a simple logger.
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.propagate = False

    return logger


class Trainable(object):
    """
    Base trainable class
    """
    abc.__metaclass__ = abc.ABCMeta

    def __init__(self):
        self.loss = None
        self.accuracy = None
        self.train_op = None
        self.session = None
        self.has_accuracy = True

        self.accumulated_loss = 0
        self.best_loss = 10e10
        self.accumulated_items = 0
        self.accumulated_valid_loss = 0
        self.accumulated_valid_accuracy = 0
        self.accumulated_valid_items = 0
        self.accumulated_accuracy = 0
        self.best_accuracy = 0

        self.session = tf.Session()

    @abc.abstractmethod
    def _create_batch_feed(self, batch, **kwargs):
        """
        Create a dictionary to feed tensorflow's placeholders
        """
        pass

    def _run_on_validation(self, data, **kwargs):
        """
        Return a tuple (accuracy, loss).

        If this problem doesn't have an accuracy definition, it should contain
        None.
        """
        self.accumulated_valid_loss = 0
        self.accumulated_valid_accuracy = 0
        self.accumulated_valid_items = 0

        def on_fetch(fetches, num_items):
            """
            Accumulate the loss and accuracy.
            """
            if self.has_accuracy:
                loss, acc = fetches
                self.accumulated_valid_accuracy += num_items * acc
            else:
                loss = fetches

            self.accumulated_valid_loss += num_items * loss
            self.accumulated_valid_items += num_items

        ops = [self.loss, self.accuracy] if self.has_accuracy else self.loss
        self._run(data, ops, 128, on_fetch, None, None)
        loss = self.accumulated_valid_loss / self.accumulated_valid_items
        if self.has_accuracy:
            acc = self.accumulated_valid_accuracy / self.accumulated_valid_items
        else:
            acc = None

        return acc, loss

    def save(self, save_dir, saver):
        pass

    def _predict(self, data, ops, batch_size, on_report=None,
                 report_interval=None):
        answers = []

        def on_fetch(fetches, _):
            answers.append(fetches)

        self._run(data, ops, batch_size, on_fetch, on_report, report_interval)
        return answers

    def _run(self, data, ops, batch_size, on_fetch, on_report,
             report_interval, **kwargs):
        """
        Run the given ops.

        :param data: input data. The model's _create_batch_feed must be able to
            create a a feed dict from these data.
        :param ops: the operators to be fetched
        :param batch_size: size of each batch
        :param on_fetch: function to be called when the required computation is
            finished and the ops are fetched. It must have the signature:
            on_fetch(fetch_data, number_of_items)
            It may be None.
        :param on_report: function to be called to report the run so far. It
            must have the signature:
            on_report(number_of_batches_run_so_far)
            It may be None.
        :param report_interval: the number of batches to be run before each call
            to on_report
        """
        batch_counter = 0
        batch_index = 0

        while batch_index < len(data):
            batch_index2 = batch_index + batch_size
            batch = data[batch_index:batch_index2]
            feeds = self._create_batch_feed(batch, **kwargs)
            fetches = self.session.run(ops, feed_dict=feeds)
            if on_fetch:
                on_fetch(fetches, len(batch))
            batch_index = batch_index2
            batch_counter += 1

            if report_interval and batch_counter % report_interval == 0:
                on_report(batch_counter)

    def train(self, train_data, dev_data, save_dir,
              num_epochs, learning_rate, batch_size, report_interval,
              dropout_keep, **kwargs):
        """
        Function with reusable training loop

        :param train_data: any data structure containing whatever training data.
            It must support indexing and len()
        :param dev_data: same as above, for training
        :param num_epochs: number of epochs to train for
        :param batch_size: size of each minibatch
        :param save_dir: directory to save model
        :param learning_rate: the learning rate
        :param report_interval: how many batches between each validation run and
            performance report
        :param dropout_keep: probability to keep values in dropout
        :param kwargs: additional training parameters used by
            self._create_batch_feed
        :return:
        """
        logger = get_logger(self.__class__.__name__)
        self.accumulated_loss = 0
        self.best_loss = 10e10
        self.accumulated_items = 0

        if self.has_accuracy:
            self.accumulated_accuracy = 0
            self.best_accuracy = 0
            train_ops = [self.train_op, self.loss, self.accuracy]
        else:
            train_ops = [self.train_op, self.loss]

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        def on_fetch(fetches, num_items):
            if self.has_accuracy:
                _, loss, accuracy = fetches
                self.accumulated_accuracy += num_items * accuracy
            else:
                _, loss = fetches

            # the number of items may be smaller than the batch size if
            # we reached the end of the dataset
            self.accumulated_items += num_items
            self.accumulated_loss += num_items * loss

        def on_report(batch_counter):
            avg_loss = self.accumulated_loss / self.accumulated_items
            if self.has_accuracy:
                avg_accuracy = self.accumulated_accuracy / \
                    self.accumulated_items
                self.accumulated_accuracy = 0

            self.accumulated_loss = 0
            self.accumulated_items = 0

            res = self._run_on_validation(dev_data, **kwargs)
            valid_acc, valid_loss = res

            msg = '%d completed epochs, %d batches' % (i, batch_counter)
            msg += '\tAvg train loss: %f' % avg_loss
            if self.has_accuracy:
                msg += '\tAvg train acc: %.2f' % avg_accuracy
            msg += '\tAvg dev loss: %f' % valid_loss
            if self.has_accuracy:
                msg += '\tAvg dev acc: %.2f' % valid_acc

            # if there is an accuracy definition, save based on it
            # otherwise, save by loss
            save_now = False
            if self.has_accuracy:
                if valid_acc > self.best_accuracy:
                    save_now = True
                    self.best_accuracy = valid_acc

            elif valid_loss < self.best_loss:
                save_now = True
                self.best_loss = valid_loss

            if save_now:
                self.save(save_dir, saver)
                msg += '\t(saved model)'

            logger.info(msg)

        for i in range(num_epochs):
            self._run(train_data, train_ops, batch_size,
                      on_fetch, on_report, report_interval,
                      learning_rate=learning_rate, dropout_keep=dropout_keep)
