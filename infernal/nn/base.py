# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import abc
import os
import math
import tensorflow as tf
import logging
from six.moves import cPickle


def print_parameters():
    """
    Count and print the number of trainable tensorflow parameters loaded in
    the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        logging.info(
            '%s: %s (%d params)' % (variable.name, shape, variable_params))
        total_params += variable_params
    return total_params


class TedinParameters(tf.contrib.training.HParams):
    """
    Subclass of tf.contrib.training.HParams holding necessary paramters for
    Tedin.
    """
    def __init__(self, learning_rate, dropout, batch_size, num_steps,
                 num_units, embeddings_shape, label_embeddings_shape,
                 num_classes, cost_regularizer=0, l2=0):
        super(TedinParameters, self).__init__(
            learning_rate=learning_rate, dropout=dropout, batch_size=batch_size,
            l2=l2, num_steps=num_steps, num_units=num_units,
            embeddings_shape=embeddings_shape,
            label_embeddings_shape=label_embeddings_shape,
            num_classes=num_classes, cost_regularizer=cost_regularizer
        )


class Trainable(object):
    """
    Base class for TEDIN
    """
    __metaclass__ = abc.ABCMeta

    # name of the file in which models are saved
    filename = None

    def __init__(self, params):
        # these attributes must be defined by subclasses
        self.logger = None
        self.session = None
        self.loss = None
        self.train_op = None
        self.params = params

    @abc.abstractmethod
    def _create_base_training_feeds(self, params):
        """
        Create the basic feeds for training, like learning rate and dropout.

        Data is not included here.
        :param params: an instance of TedinParameters
        :return: dictionary
        """
        return {}

    @abc.abstractmethod
    def _get_next_batch(self, data, batch_size, training):
        return data

    @property
    def train_fetches(self):
        """
        Tensorflow fetches to be computed during each training step.

        Defaults to self.loss and self.train_op
        """
        return [self.loss, self.train_op]

    @property
    def validation_fetches(self):
        return [self.loss]

    def _run(self, feeds, fetches, data):
        """
        Run the model for the given data.

        :param feeds: Extra feeds with dropout, regularizer etc if applicable
        :param fetches: Tensors to fetch
        :param data: The input data
        :return: List with computed fecthes
        """
        data_feeds = self._create_data_feeds(data)
        data_feeds.update(feeds)

        return self.session.run(fetches, data_feeds)

    @abc.abstractmethod
    def _create_data_feeds(self, data):
        """
        Create a feed dictionary with the given data.
        """
        return {}

    @abc.abstractmethod
    def _init_train_stats(self, params, report_interval):
        """
        Initialize some training statistics.
        """
        pass

    def _init_validation(self, data):
        """
        Initialize any performance counters and reset the data batch counter
        """
        data.reset_batch_counter()

    @abc.abstractmethod
    def _update_training_stats(self, values, data):
        """
        Update training statistics such as accuracy and loss.

        :param values: list of computed values of train_fetches
        """
        pass

    @abc.abstractmethod
    def _update_validation_stats(self, values, data):
        """
        Update training statistics such as accuracy and loss.

        :param values: list of computed values of train_fetches
        """
        pass

    @abc.abstractmethod
    def _get_validation_metrics(self, data):
        """
        Compute the final validation metrics such as accuracy and loss
        """
        pass

    def _get_data_size(self, data):
        """
        Determine the size of the given input data
        """
        return len(data)

    @abc.abstractmethod
    def _validation_report(self, values, saver, step, train_data, model_dir):
        """
        Report perform and possibly save a model
        """
        pass

    def _run_validation(self, data, batch_size, saver, train_step, train_data,
                        model_dir):
        """
        Run the model on validation data

        :param data: Dataset
        :param batch_size: None or integer. If None, the whole dataset will be
            evaluated at once, which may not fit memory.
        :return: list with validation_fetches
        """
        data_size = self._get_data_size(data)
        if batch_size is None or batch_size > data_size:
            batch_size = data_size

        num_batches = math.ceil(data_size / batch_size)
        self._init_validation(data)
        for _ in range(num_batches):
            batch = self._get_next_batch(data, batch_size, training=False)
            values = self._run({}, self.validation_fetches, batch)

            # multiply by len(batch) because the last batch may have a different
            # length
            self._update_validation_stats(values, batch)

        metrics = self._get_validation_metrics(data)
        self._validation_report(metrics, saver, train_step, train_data,
                                model_dir)

    @classmethod
    def get_base_file_name(cls, path):
        """
        Return the base filename used for serializing stuff
        """
        return os.path.join(path, cls.filename)

    def save_params(self):
        """
        Save the model hyperparameters
        """
        #TODO figure out a way to save and restore HParams as protocol buffers
        # (it is totally undocumented)
        filename = self.path + '-hparams.pickle'
        with open(filename, 'wb') as f:
            cPickle.dump(self.params, f)

    @classmethod
    def load_params(cls, directory):
        """
        Load the model hyperparamters from the given directory
        """
        path = os.path.join(directory, cls.filename) + '-hparams.pickle'
        with open(path, 'rb') as f:
            params = cPickle.load(f)
        return params

    def train(self, train_data, valid_data, model_dir, report_interval):
        """
        Train the model

        :param train_data: tuple of Datasets (positive, negative)
        :param valid_data: tuple of Datasets (positive, negative)
        :param model_dir: path to the model dir
        :param report_interval: how many steps before each performance report
        :return:
        """
        self._init_train_stats(self.params, report_interval)

        self.path = os.path.join(model_dir, self.filename)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        self.save_params()

        feeds = self._create_base_training_feeds(self.params)

        self.logger.info('Starting training')
        for step in range(1, self.params.num_steps + 1):
            batch = self._get_next_batch(train_data, self.params.batch_size,
                                         training=True)
            fetches = self.train_fetches
            values = self._run(feeds, fetches, batch)
            self._update_training_stats(values, batch)

            if step % report_interval == 0:
                self._run_validation(valid_data, 256, saver, step,
                                     train_data, model_dir)
