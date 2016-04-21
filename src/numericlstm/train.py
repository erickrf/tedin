# -*- coding: utf-8 -*-

"""
Scripts for training the numeric LSTM.
"""

import tensorflow as tf
import numpy as np
import logging

import utils
from utils import Symbol
import memorizer
import config


total_data = config.train_size + config.valid_size
data, sizes = utils.generate_dataset(config.num_time_steps, total_data)
utils.shuffle_data_and_sizes(data, sizes)

# removing duplicates must be change to account for the sizes.... at any rate,
# we were getting 5 duplicates out of 32k. i don't think we really need it
#data = remove_duplicates(data)
train_set = data[:, :config.train_size]
valid_set = data[:, config.train_size:]
train_sizes = sizes[:config.train_size]
valid_sizes = sizes[config.train_size:]

n_train = train_set.shape[1]
n_valid = valid_set.shape[1]
num_batches = int(n_train / config.batch_size)
print 'Training with %d sequences; %d for validation' % (n_train, n_valid)

sess = tf.Session()
model = memorizer.NumericLSTM(config.embedding_size, config.num_time_steps)

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

logging.info('Initialized the model and all variables. Starting train...')


def get_accuracy(model, session, data, sizes, ignore_end=True):
    """
    Get the prediciton accuracy on the supplied data.

    :param model: an instance of the numeric LSTM
    :param session: current tensorflow session
    :param data: numpy array with shape (num_time_steps, batch_size)
    :param sizes: actual size of each sequence in data
    :param ignore_end: if True, ignore the END symbol
    """
    answer = model.run(session, data, sizes)

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


for epoch_num in range(config.num_epochs):
    # loop for epochs - each one goes through the whole dataset
    utils.shuffle_data_and_sizes(train_set, train_sizes)
    last_batch_idx = 0

    for batch_num in range(num_batches):
        batch_idx = last_batch_idx + config.batch_size
        batch = train_set[:, last_batch_idx:batch_idx]
        sizes = train_sizes[last_batch_idx:batch_idx]
        last_batch_idx = batch_idx

        feeds = {model.first_term: batch,
                 model.first_term_size: sizes,
                 model.l2_constant: 0.0001,
                 model.learning_rate: 0.1}

        _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feeds)
        accumulated_loss += loss_value

        if (batch_num + 1) % config.report_interval == 0:
            print('Epoch %d, batch %d' % (epoch_num + 1, batch_num + 1))
            avg_loss = accumulated_loss / config.report_interval
            print('Train loss: %.5f' % avg_loss)
            accumulated_loss = 0

            valid_acc = get_accuracy(model, sess, valid_set, valid_sizes)
            print('Validation accuracy: %f' % valid_acc)

    print('')
