# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Script for training the memorizer autoencoder.
"""

import tensorflow as tf
import logging
import argparse

import utils
import datageneration
import memorizer
import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('save_file', help='Path to file to save trained model')
    parser.add_argument('-t', help='Maximum number of time steps', default=9,
                        dest='num_time_steps', type=int)
    parser.add_argument('-n', help='Embedding size', default=300,
                        dest='embedding_size', type=int)
    parser.add_argument('-r', help='Initial learning rate', default=0.1, dest='learning_rate',
                        type=float)
    parser.add_argument('-b', help='Batch size', default=32, dest='batch_size', type=int)
    parser.add_argument('-e', help='Number of epochs', default=10, dest='num_epochs', type=int)
    parser.add_argument('--train', help='Size of the training set', default=32000, type=int)
    parser.add_argument('--valid', help='Size of the validation set', default=2000, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    datasets = datageneration.generate_memorizer_data(args.train,
                                                      args.valid,
                                                      args.num_time_steps)
    train_set, train_sizes, valid_set, valid_sizes = datasets

    num_batches = int(args.train / args.batch_size)
    logging.info('Training with %d sequences; %d for validation' % (args.train, args.valid))

    sess = tf.Session()
    model = memorizer.MemorizerAutoEncoder(args.embedding_size, args.num_time_steps,
                                           args.learning_rate)

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    utils.save_parameters(args.save_file, args.embedding_size, args.num_time_steps)
    logging.info('Initialized the model and all variables. Starting training...')

    best_acc = 0
    accumulated_loss = 0
    for epoch_num in range(args.num_epochs):
        # loop for epochs - each one goes through the whole dataset
        utils.shuffle_data_memorizer(train_set, train_sizes)
        last_batch_idx = 0

        for batch_num in range(num_batches):
            batch_idx = last_batch_idx + args.batch_size
            batch = train_set[:, last_batch_idx:batch_idx]
            sizes = train_sizes[last_batch_idx:batch_idx]
            last_batch_idx = batch_idx

            feeds = {model.first_term: batch,
                     model.first_term_size: sizes,
                     model.l2_constant: 0.0001}

            _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feeds)
            accumulated_loss += loss_value

            if (batch_num + 1) % config.report_interval == 0:
                avg_loss = accumulated_loss / config.report_interval
                valid_answers = model.run(sess, valid_set, valid_sizes)
                valid_acc = utils.compute_accuracy(valid_set, valid_answers)

                logging.info('Epoch %d, batch %d' % (epoch_num + 1, batch_num + 1))
                logging.info('Train loss: %.5f' % avg_loss)

                msg = 'Validation accuracy: %f' % valid_acc
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    saver.save(sess, args.save_file)
                    msg += ' (new model saved)'
                logging.info(msg)

                accumulated_loss = 0

        print()
