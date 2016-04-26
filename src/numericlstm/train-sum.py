# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Script for training the addition autoencoder.
"""

import tensorflow as tf
import argparse
import logging

import utils
import addition
import config
import datageneration as dg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('save_file', help='Path to file to save trained model')
    parser.add_argument('-l', dest='load_file',
                        help='Path to file with pre-trained parameters '
                             '(if given, ignores the embedding size parameter)')
    parser.add_argument('-t', help='Maximum number of time steps', default=9,
                        dest='num_time_steps', type=int)
    parser.add_argument('-n', help='Embedding size', default=300,
                        dest='embedding_size', type=int)
    parser.add_argument('-b', help='Batch size', default=32, dest='batch_size', type=int)
    parser.add_argument('-e', help='Number of epochs', default=10, dest='num_epochs', type=int)
    parser.add_argument('--train', help='Size of the training set', default=32000, type=int)
    parser.add_argument('--valid', help='Size of the validation set', default=2000, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.load_file is not None:
        params = utils.load_parameters(args.load)
        num_time_steps = params['num_time_steps']
        embedding_size = params['embedding_size']
    else:
        num_time_steps = args.num_time_steps
        embedding_size = args.embedding_size

    train_data = dg.generate_addition_data(num_time_steps, args.train)
    first_terms, second_terms, first_sizes, second_sizes, results = train_data

    num_batches = int(args.train / args.batch_size)
    logging.info('Training with %d sequences; %d for validation' % (args.train, args.valid))

    sess = tf.Session()
    model = addition.AdditionAutoEncoder(embedding_size, num_time_steps)

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    utils.save_parameters(args.save_file, embedding_size, num_time_steps)
    logging.info('Initialized the model and all variables. Starting training...')

    best_acc = 0
    accumulated_loss = 0
    for epoch_num in range(args.num_epochs):
        # loop for epochs - each one goes through the whole dataset
        utils.shuffle_data_addition(first_terms, second_terms,
                                    first_sizes, second_sizes,
                                    results)
        last_batch_idx = 0

        for batch_num in range(num_batches):
            batch_idx = last_batch_idx + args.batch_size

            batch_1st_term = first_terms[:, last_batch_idx:batch_idx]
            batch_2nd_term = second_terms[:, last_batch_idx:batch_idx]
            batch_1st_size = first_sizes[last_batch_idx:batch_idx]
            batch_2nd_size = second_sizes[last_batch_idx:batch_idx]
            batch_result = results[:, last_batch_idx:batch_idx]

            last_batch_idx = batch_idx

            feeds = {model.first_term: batch_1st_term,
                     model.first_term_size: batch_1st_size,
                     model.second_term: batch_2nd_term,
                     model.second_term_size: batch_2nd_size,
                     model.sum: batch_result,
                     model.l2_constant: 0.0001,
                     model.learning_rate: 0.1}

            _, loss_value = sess.run([model.train_op, model.loss], feed_dict=feeds)
            accumulated_loss += loss_value

            if (batch_num + 1) % config.report_interval == 0:
                avg_loss = accumulated_loss / config.report_interval

                logging.info('Epoch %d, batch %d' % (epoch_num + 1, batch_num + 1))
                logging.info('Train loss: %.5f' % avg_loss)

                accumulated_loss = 0

        print()


