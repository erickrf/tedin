# -*- coding: utf-8 -*-

"""
Class for a model that learns weights for different tree edit operations.

The treee edit distance code was adapted from the Python module zss.
"""

import zss
import numpy as np
import tensorflow as tf
import os

from ..datastructures import Dataset
from .. import utils

# operation codes
INSERT = zss.Operation.insert
REMOVE = zss.Operation.remove
UPDATE = zss.Operation.update
MATCH = zss.Operation.match


class TedinParameters(tf.contrib.training.HParams):
    """
    Subclass of tf.contrib.training.HParams holding necessary paramters for
    Tedin.
    """
    def __init__(self, learning_rate, dropout, batch_size, num_steps, l2=0,
                 gradient_clipping=None):
        super(TedinParameters, self).__init__()
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.l2 = l2
        self.gradient_clipping = gradient_clipping
        self.num_steps = num_steps


def find_zss_operations(pair, insert_costs, remove_costs, update_cost_fn):
    """
    Run the zhang-shasha algorithm and return the list of operations

    :param pair: an instance of Pair
    :param insert_costs: array with costs of inserting each node in sentence 2
    :param remove_costs: array with costs of removing each node in sentence 1
    :param update_cost_fn: function (node1, node2) -> update cost
    :return:
    """
    def get_children(node):
        return node.dependents

    def get_insert_cost(node):
        return insert_costs[node.id - 1]

    def get_remove_cost(node):
        return remove_costs[node.id - 1]

    tree_t = pair.annotated_t
    tree_h = pair.annotated_h
    root_t = tree_t.root
    root_h = tree_h.root

    _, ops = zss.distance(root_t, root_h, get_children, get_insert_cost,
                          get_remove_cost, update_cost_fn, lambda _: None)
    return ops


def create_tedin_dataset(pairs, wd, lower=True):
    """
    Create a Dataset object to feed a Tedin model.

    :param pairs: list of parsed Pair objects
    :param wd: word dictionary mapping tokens to integers
    :param lower: whether to lowercase tokens
    :return: Dataset
    """
    def index(token):
        if lower:
            return wd[token.lower()]
        return wd[token]

    nodes1 = []
    nodes2 = []
    labels = []
    for pair in pairs:
        t = pair.annotated_t
        h = pair.annotated_h
        nodes1.append([index(token.text) for token in t.tokens])
        nodes2.append([index(token.text) for token in h.tokens])
        labels.append(pair.entailment.value)

    nodes1, sizes1 = utils.nested_list_to_array(nodes1)
    nodes2, sizes2 = utils.nested_list_to_array(nodes2)
    # labels are originally numbered starting from 1
    labels = np.array(labels) - 1
    dataset = Dataset(pairs, nodes1, nodes2, sizes1, sizes2, labels)

    return dataset


class TreeEditDistanceNetwork(object):
    """
    Model that learns weights for different tree edit operations.
    """
    filename = 'tedin'

    def __init__(self, num_hidden_units, embeddings_shape, num_classes):
        """

        :param num_hidden_units: number of units in hidden layers
        """
        self.session = tf.Session()
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes
        self.logger = utils.get_logger(self.__class__.__name__)

        # hyperparameters
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
        self.dropout_keep = tf.placeholder_with_default(
            1., None, 'dropout_keep')

        self.embedding_ph = tf.placeholder(tf.float32, embeddings_shape,
                                           'word_embeddings_ph')

        # labels for the supervised training
        self.labels = tf.placeholder(tf.int32, [None], 'labels')

        # operations applied to a sentence pair (batch, max_num_ops)
        self.operations = tf.placeholder(tf.int32, [None, None], 'operations')

        self.embeddings = tf.Variable(self.embedding_ph, trainable=False,
                                      validate_shape=True, name='embeddings')

        self.update_cache = {}

        # control weights already initialized
        self._initialized_weights = set()
        self._define_operation_costs()
        self._define_supervised_classifier()

    def _define_supervised_classifier(self):
        """
        Define the tensors related to the supervised classification of pairs
        """
        # arguments of operations; shape is always (batch, max_num_ops)
        # each batch item has all the nodes (word indices) given as argument
        # for that operation in that pair
        self.args_insert = tf.placeholder(tf.int32, [None, None], 'args_insert')
        self.args_remove = tf.placeholder(tf.int32, [None, None], 'args_remove')
        self.args1_update = tf.placeholder(tf.int32, [None, None],
                                           'args1_update')
        self.args2_update = tf.placeholder(tf.int32, [None, None],
                                           'args2_update')

        # these are the lengths of the sequences of operations
        self.num_inserts = tf.placeholder(tf.int32, [None], 'num_inserts')
        self.num_removes = tf.placeholder(tf.int32, [None], 'num_inserts')
        self.num_updates = tf.placeholder(tf.int32, [None], 'num_updates')

        # these embedded values are (batch, max_ops, embedding_size)
        emb_insert = tf.nn.embedding_lookup(self.embeddings, self.args_insert)
        emb_remove = tf.nn.embedding_lookup(self.embeddings, self.args_remove)
        emb_update1 = tf.nn.embedding_lookup(self.embeddings, self.args1_update)
        emb_update2 = tf.nn.embedding_lookup(self.embeddings, self.args2_update)

        # all representations are (batch, max_ops, hidden_units)
        rep_insert = self.get_operation_representation('insert', emb_insert)
        rep_remove = self.get_operation_representation('remove', emb_remove)
        rep_update = self.get_operation_representation('update', emb_update1,
                                                       emb_update2)

        # run separate convolutions on the 3 sequences and then take one single
        # maximum. this is because joining them could be harder.
        conv_insert = self._convolution(rep_insert, self.num_inserts)
        conv_remove = self._convolution(rep_remove, self.num_removes, True)
        conv_update = self._convolution(rep_update, self.num_updates, True)

        with tf.variable_scope('convolution'):
            tensor_list = [conv_insert, conv_remove, conv_update]
            max_values = tf.reduce_max(tensor_list, 0)
            conv_output = tf.nn.dropout(max_values, self.dropout_keep)

        with tf.variable_scope('softmax'):
            init = tf.glorot_normal_initializer()
            logits = tf.layers.dense(conv_output, self.num_classes,
                                     kernel_initializer=init)

        self.answers = tf.argmax(logits, 1, 'answers')

        # this is the moving accuracy, should be used on the training set
        self.moving_acc, self.update_acc_op = tf.metrics.accuracy(self.labels,
                                                                  self.answers)
        # this is a static accuracy, evaluated once on the validation set
        hits = tf.equal(tf.cast(self.answers, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

        cross_ent = tf.losses.sparse_softmax_cross_entropy(self.labels, logits)
        self.loss = tf.reduce_mean(cross_ent, name='loss')

        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def _convolution(self, inputs, lengths, reuse=False):
        """
        Apply a convolution to the inputs and return the max over time.

        In practice, the convolution is implemented as a dense layer, since it
        is 1-dimensional with step 1 and kernel size 1 (inputs are operations,
        understood to be unordered).

        :param inputs: tensor (batch, time_steps, num_units)
        :return: tensor (batch, num_units)
        """
        with tf.variable_scope('convolution', reuse=reuse):
            hidden = tf.layers.dense(inputs, self.num_hidden_units, tf.nn.relu)

            # mask positions after sequence end with -inf
            max_length = tf.shape(inputs)[1]
            mask = tf.sequence_mask(lengths, max_length, name='mask')
            mask3d = tf.tile(tf.expand_dims(mask, 2),
                             [1, 1, self.num_hidden_units])
            masked = tf.where(mask3d, hidden, -np.inf * tf.ones_like(hidden))
            max_values = tf.reduce_max(masked, 1)

        return max_values

    def _define_operation_costs(self):
        """
        Define the tensors related to the cost of tree edit operations
        """
        # nodes are the tokens; shape is (batch, sentence_length)
        self.nodes1 = tf.placeholder(tf.int32, [None, None], 'nodes1')
        self.nodes2 = tf.placeholder(tf.int32, [None, None], 'nodes2')

        # single nodes are used for computing update costs for a node pair
        self.single_node1 = tf.placeholder(tf.int32, [None], 'single_node1')
        self.single_node2 = tf.placeholder(tf.int32, [None], 'single_node2')

        embedded1 = tf.nn.embedding_lookup(self.embeddings, self.nodes1)
        embedded2 = tf.nn.embedding_lookup(self.embeddings, self.nodes2)
        self.remove_costs = self._compute_operation_cost('remove', embedded1)
        self.insert_costs = self._compute_operation_cost('insert', embedded2)

        # computing all node1/node2 update combinations is too expensive and
        # unnecessary. Instead, call it as needed with a pair of nodes.
        self.update_cache = {}
        single_embedded1 = tf.nn.embedding_lookup(self.embeddings,
                                                  self.single_node1)
        single_embedded2 = tf.nn.embedding_lookup(self.embeddings,
                                                  self.single_node2)
        self.update_cost = self._compute_operation_cost(
            'update', single_embedded1, single_embedded2)

    def classify_operation_sequence(self):
        pass

    def _compute_operation_cost(self, scope, node1, node2=None):
        """
        Apply a two-layer transformation to the input nodes yielding an
        operation cost.

        :param scope: name of the scope ('insert', 'remove' or 'update')
        :param node1: tensor 2+d with embedded nodes, shape
            (batch, ..., num_units)
        :param node2: tensor 2+d with embedded nodes, shape
            (batch, ..., num_units)
            (only used in update operations)
        :return: tensor with the shape of node1 without the last dimension:
            (batch, ...)
        """
        op_representation = self.get_operation_representation(scope, node1,
                                                              node2)

        scope = 'operation_cost'
        reuse = scope in self._initialized_weights
        self._initialized_weights.add(scope)
        with tf.variable_scope(scope, reuse=reuse):
            init = tf.glorot_normal_initializer()
            hidden = tf.layers.dense(op_representation, self.num_hidden_units,
                                     tf.nn.relu, kernel_initializer=init)
            hidden = tf.nn.dropout(hidden, self.dropout_keep)

            # init cost bias to 1 so that it is centered around 1 instead of 0
            # since costs can't be negative
            cost = tf.layers.dense(hidden, 1, tf.nn.relu,
                                   bias_initializer=tf.ones_initializer(),
                                   kernel_initializer=init)
            cost = tf.reshape(cost, tf.shape(node1)[:-1])

        return cost

    def get_operation_representation(self, operation, node1, node2=None):
        """
        Compute the representation of an operation and involved arguments.

        The nodes are processed by a one-layer network whose weights depend
        on the operation type.

        Note that the same operation is applied to all items in the batch. In
        order to decide

        :param operation: 'insert', 'remove' or 'update'
        :param node1: embedded node, shape (batch, num_nodes, num_units)
        :param node2: embedded node, shape (batch, num_nodes, num_units)
            (only used in update operations)
        :return: tensor shape (batch, num_nodes, num_hidden_units)
        """
        if node2 is None:
            inputs = node1
        else:
            inputs = tf.concat([node1, node2], axis=-1)

        reuse = operation in self._initialized_weights
        self._initialized_weights.add(operation)
        with tf.variable_scope(operation, reuse=reuse):
            hidden = tf.layers.dense(inputs, self.num_hidden_units, tf.nn.relu)
            hidden = tf.nn.dropout(hidden, self.dropout_keep)

        return hidden

    def create_operation_feeds(self, batch):
        """
        Run the zhang-shasha algorithm to find the sequence of edit operations.

        :param batch: Dataset
        :return: dictionary of feeds with the operations and their arguments
        """
        def run_update_cost(node1, node2):
            """
            Compute the update cost for a single node substitution

            :param node1: Token object
            :param node2: Token object
            :return: int
            """
            # take the embedding index of each word
            id1 = node1.index
            id2 = node2.index
            if (node1, node2) in self.update_cache:
                return self.update_cache[(id1, id2)]

            feeds = {self.single_node1: [id1], self.single_node2: [id2]}
            cost = self.update_cost.eval(feeds, session=self.session)[0]
            self.update_cache[(node1, node2)] = cost

            return cost

        # precompute the insert and remove costs
        feeds = {self.nodes1: batch.nodes1, self.nodes2: batch.nodes2}
        insert_costs, remove_costs = self.session.run([self.insert_costs,
                                                       self.remove_costs],
                                                      feeds)
        self.update_cache = {}

        all_insert_args = []
        all_remove_args = []
        all_update_args1 = []
        all_update_args2 = []
        for i, item in enumerate(batch):
            operations = find_zss_operations(
                item.pairs, insert_costs[i], remove_costs[i], run_update_cost)

            inserts = []
            removes = []
            updates1 = []
            updates2 = []
            for op in operations:
                if op.type == INSERT:
                    inserts.append(op.arg2.index)
                elif op.type == REMOVE:
                    removes.append(op.arg1.index)
                elif op.type == UPDATE:
                    updates1.append(op.arg1.index)
                    updates2.append(op.arg2.index)

            all_insert_args.append(inserts)
            all_remove_args.append(removes)
            all_update_args1.append(updates1)
            all_update_args2.append(updates2)

        try:
            insert_args, num_inserts = utils.nested_list_to_array(
                all_insert_args)
        except ValueError:
            print(len(batch))
            print(all_insert_args)
            print(all_remove_args)
            print(all_update_args1)
            raise

        remove_args, num_removes = utils.nested_list_to_array(
            all_remove_args)
        update_args1, num_updates = utils.nested_list_to_array(
            all_update_args1)
        update_args2, _ = utils.nested_list_to_array(
            all_update_args2)

        feeds = {self.args_insert: insert_args, self.num_inserts: num_inserts,
                 self.args_remove: remove_args, self.num_removes: num_removes,
                 self.args1_update: update_args1, self.num_updates: num_updates,
                 self.args2_update: update_args2}

        return feeds

    def initialize(self, embeddings):
        """
        Initialize trainable variables and embeddings.

        :param embeddings: numpy array matching the shape given to the
            constructor
        """
        self.logger.debug('Initializing variables')
        feed = {self.embedding_ph: embeddings}
        self.session.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()], feed)

    def train(self, train_data, valid_data, params, model_dir, report_interval):
        """
        Train the network

        :param train_data: Dataset
        :type train_data: Dataset
        :param valid_data: Dataset
        :type valid_data: Dataset
        :param params: TedinParameters
        :type params: TedinParameters
        :param report_interval: number of steps between validation run and
            performance reports
        :param model_dir: path to save the trained model
        :return:
        """
        best_acc = 0
        accumulated_training_loss = 0
        loss_denominator = params.batch_size * report_interval
        train_params_feeds = {self.learning_rate: params.learning_rate,
                              self.dropout_keep: params.dropout}

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        filename = os.path.join(model_dir, self.filename)

        self.logger.info('Starting training')
        # count steps from 1 to make it easier to treat `report_interval`
        for step in range(1, params.num_steps):
            batch = train_data.next_batch(params.batch_size, wrap=True)
            train_params_feeds.update({self.labels: batch.labels})

            ops = [self.loss, self.update_acc_op, self.train_op]
            loss, _, _ = self._run(batch, ops, train_params_feeds)
            accumulated_training_loss += loss

            if step % report_interval == 0:
                train_acc = self.session.run(self.moving_acc)
                train_loss = accumulated_training_loss / loss_denominator
                self._reset_metrics()

                valid_acc, valid_loss = self.run_validation(valid_data)
                msg = '{} epochs\t{} steps\tTrain acc: {:.5}\t' \
                      'Train loss: {:.5}\tValid acc: {:.5}\tValid loss: {:.5}'
                if valid_acc > best_acc:
                    saver.save(self.session, filename, step)
                    best_acc = valid_acc
                    msg += ' (saved model)'

                self.logger.info(msg.format(train_data.epoch, step, train_acc,
                                            train_loss, valid_acc, valid_loss))

    def _reset_metrics(self):
        """
        Reset performance metrics
        """
        metric_vars = [v for v in tf.local_variables()
                       if 'accuracy' in v.name]
        self.session.run(tf.variables_initializer(metric_vars))

    def _run(self, data, fetches, extra_feeds):
        """
        Run the model for the given data.

        It computes operation costs, runs zhang-shasha in python to get the
        operation sequence, and the run the convolution over the operations.

        :param data: dataset
        :param extra_feeds: any extra feed data to the model
        :return: the evaluated fetches
        """
        op_feeds = self.create_operation_feeds(data)
        op_feeds.update(extra_feeds)

        return self.session.run(fetches, op_feeds)

    def run_validation(self, data, batch_size=None):
        """
        Run the model on validation data and return the accuracy and loss.

        :param data: Dataset
        :param batch_size: None or integer. If None, the whole dataset will be
            evaluated at once, which may not fit memory.
        :return: tuple (accuracy, loss) as python floats
        """
        if batch_size is None:
            batch_size = len(data)

        accumulated_loss = 0
        accumulated_acc = 0
        num_batches = int(len(data) / batch_size)
        data.reset_batch_counter()
        for _ in range(num_batches):
            batch = data.next_batch(batch_size, wrap=False)
            acc, loss = self._run(batch, [self.accuracy, self.loss],
                                  {self.labels: batch.labels})

            # multiply by len(batch) because the last batch may have a different
            # length
            accumulated_acc += acc * len(batch)
            accumulated_loss += loss * len(batch)

        acc = accumulated_acc / len(data)
        loss = accumulated_loss / len(data)

        return acc, loss
