# -*- coding: utf-8 -*-

"""
Scripts for training the numeric LSTM.
"""

import tensorflow as tf
import numpy as np

import utils
from utils import Symbol
import memorizer

num_time_steps = 9
embedding_size = 300

# get the data
train_size = 32000
valid_size = 1000

total_data = train_size + valid_size
data, sizes = utils.generate_dataset(num_time_steps, total_data)
utils.shuffle_data_and_sizes(data, sizes)

# removing duplicates must be change to account for the sizes.... at any rate,
# we were getting 5 duplicates out of 32k. i don't think we really need it
#data = remove_duplicates(data)
train_set = data[:, :train_size]
valid_set = data[:, train_size:]
train_sizes = sizes[:train_size]
valid_sizes = sizes[train_size:]

n_train = train_set.shape[1]
n_valid = valid_set.shape[1]
print 'Training with %d sequences; %d for validation' % (n_train, n_valid)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_size = 32
num_epochs = 1

accumulated_loss = 0
report_interval = 100
save_path = '../checkpoints/basic-memorizer.dat'

num_batches = int(n_train / batch_size)

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
model = memorizer.NumericLSTM(embedding_size, num_time_steps)


def get_accuracy(model, data, sizes, ignore_end=True):
    """
    Get the prediciton accuracy on the supplied data.

    :param ignore_end: if True, ignore the END symbol
    """
    answer = model.run_network(data, sizes)

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


for epoch_num in range(num_epochs):
    # loop for epochs - each one goes through the whole dataset
    utils.shuffle_data_and_sizes(train_set, train_sizes)
    last_batch_idx = 0

    for batch_num in range(num_batches):
        batch_idx = last_batch_idx + batch_size
        batch = train_set[:, last_batch_idx:batch_idx]
        sizes = train_sizes[last_batch_idx:batch_idx]
        last_batch_idx = batch_idx

        feeds = {model.first_term: batch,
                 model.first_term_size: sizes,
                 model.l2_constant: 0.0001,
                 model.learning_rate: 0.1}

        _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feeds)
        accumulated_loss += loss_value

        if (batch_num + 1) % report_interval == 0:
            print('Epoch %d, batch %d' % (epoch_num + 1, batch_num + 1))
            avg_loss = accumulated_loss / report_interval
            print('Train loss: %.5f' % avg_loss)
            accumulated_loss = 0

            valid_acc = get_accuracy(model, valid_set, valid_sizes)
            print('Validation accuracy: %f' % valid_acc)

    print('')
