# -*- coding: utf-8 -*-

"""
Class for a model that learns weights for different tree edit operations.

The treee edit distance code was adapted from the Python module zss.
"""

import zss
import numpy as np
import tensorflow as tf

from .base import TedinParameters, Trainable
from ..datastructures import Dataset, Token
from .. import utils

# operation codes
INSERT = zss.Operation.insert
REMOVE = zss.Operation.remove
UPDATE = zss.Operation.update
MATCH = zss.Operation.match


def get_variable_by_name(name):
    return [v for v in tf.global_variables() if v.name == name][0]


def find_zss_operations(pair, insert_costs, remove_costs, update_costs):
    """
    Run the zhang-shasha algorithm and return the list of operations

    :param pair: an instance of Pair
    :param insert_costs: array with costs of inserting each node in sentence 2
    :param remove_costs: array with costs of removing each node in sentence 1
    :param update_costs: array 2d with costs of replacing node i from sentence 1
        with node j from sentence 2 in cell (i, j)
    :return:
    """
    def get_children(node):
        return node.dependents

    def get_insert_cost(node):
        return insert_costs[node.id - 1]

    def get_remove_cost(node):
        return remove_costs[node.id - 1]

    def get_update_cost(node1, node2):
        if node1.index == node2.index and node1.dep_index == node2.dep_index:
            return 0
        return update_costs[node1.id - 1, node2.id - 1]

    tree_t = pair.annotated_t
    tree_h = pair.annotated_h
    root_t = tree_t.root
    root_h = tree_h.root

    cost, ops = zss.distance(root_t, root_h, get_children, get_insert_cost,
                             get_remove_cost, get_update_cost,
                             return_operations=True)
    return cost, ops


def mask_2d(inputs, lengths, mask_value):
    """
    Mask the values past sequence length

    :param inputs: tensor (batch, time_steps, units)
    :param lengths: tensor (batch) with the real length (non-padding) of each
        item
    :param mask_value: mask value
    :return: masked values
    """
    max_length = tf.shape(inputs)[1]
    mask = tf.sequence_mask(lengths, max_length, name='mask')
    masked = tf.where(mask, inputs, mask_value * tf.ones_like(inputs))

    return masked


def mask_3d(inputs, lengths, mask_value):
    """
    Mask the values past sequence length

    :param inputs: tensor (batch, time_steps, units)
    :param lengths: tensor (batch) with the real length (non-padding) of each
        item
    :param mask_value: mask value
    :return: masked values
    """
    max_length = tf.shape(inputs)[1]
    num_units = tf.shape(inputs)[2]
    mask = tf.sequence_mask(lengths, max_length, name='mask')
    mask3d = tf.tile(tf.expand_dims(mask, 2),
                     [1, 1, num_units])
    masked = tf.where(mask3d, inputs, mask_value * tf.ones_like(inputs))

    return masked


class TreeEditDistanceNetwork(Trainable):
    """
    Model that learns weights for different tree edit operations.
    """
    filename = 'tedin'

    def __init__(self, params, session=None, word_embeddings_var=None,
                 reuse_weights=False, create_optimizer=True):
        """
        :param params: TedinParameters
        :type params: TedinParameters
        :param session: tensorflow session or None. If None, a new session is
            created, otherwise the supplied one is used.
        :param reuse_weights: if True, all weights will be reused.
        :param word_embeddings_var: tensorflow variable or None.
            This should NOT be a numpy array with the embeddings; they are
            provided to the initialize method.
            It should be used when sharing the embeddings with another model.
        :param create_optimizer: if True, create an optimizer for the labeled
            problem. It should be false when using two instances in tandem for
            the ranking task.
        """
        if session is None:
            session = tf.Session()

        self.session = session
        self.params = params
        self.num_hidden_units = params.num_units
        self.num_classes = params.num_classes
        self.logger = utils.get_logger(self.__class__.__name__)

        # hyperparameters
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')
        self.dropout_keep = tf.placeholder_with_default(
            1., None, 'dropout_keep')
        self.cost_regularizer = tf.placeholder_with_default(
            0., None, 'cost_regularizer')

        # labels for the supervised training
        self.labels = tf.placeholder(tf.int32, [None], 'labels')

        # operations applied to a sentence pair (batch, max_num_ops)
        self.operations = tf.placeholder(tf.int32, [None, None], 'operations')

        if word_embeddings_var is None:
            self.embedding_ph = tf.placeholder(
                tf.float32, params.embeddings_shape, 'word_embeddings_ph')
            self.embeddings = tf.Variable(
                self.embedding_ph, trainable=False, validate_shape=True,
                name='word_embeddings')
        else:
            self.embeddings = word_embeddings_var

        with tf.variable_scope('attribute-embedding', reuse=reuse_weights):
            # "label" here refers to the node labels, not the target class
            self.label_embeddings = tf.get_variable(
                'dependency_embeddings', params.label_embeddings_shape,
                tf.float32, tf.random_normal_initializer(0, 0.1))

        # control weights already initialized
        self.reuse_weights = reuse_weights
        self._initialized_weights = set()
        self._define_operation_costs()
        self._define_computation_graph(create_optimizer)

    @classmethod
    def load(cls, path, embeddings, session=None):
        """
        Load a saved model

        :param path: directory with saved model
        :param embeddings: numpy embedding matrix
        :param session: existing tensorflow session to be reuse, or None to
            create a new one
        :return: TEDIN instance
        """
        if session is None:
            session = tf.Session()

        params = cls.load_params(path)
        tedin = TreeEditDistanceNetwork(params, session)
        tedin.initialize(embeddings)

        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(session, cls.get_base_file_name(path))

        return tedin

    def _create_base_training_feeds(self, params):
        feeds = {self.learning_rate: params.learning_rate,
                 self.dropout_keep: params.dropout}
        return feeds

    def _define_transformation_cost(self):
        """
        Define the tensors related to the total transformation cost.

        This is used when training the model for pairwise ranking.
        """
        def sum_real_operation_costs(scope, args1, args2, length):
            """
            Sum the costs of the real (non-padding) operations
            """
            # padded_costs is (batch, num_operations)
            padded_costs = self._define_operation_cost(scope, args1, args2)

            # mask to zero the values of padding operations
            real_costs = mask_2d(padded_costs, length, 0)

            # total_costs is (batch,)
            total_costs = tf.reduce_sum(real_costs, 1, name='total_costs')

            return total_costs

        # each of these costs variable has shape (batch,)
        costs_insert = sum_real_operation_costs('insert', self.emb_insert,
                                                None, self.num_inserts)
        costs_remove = sum_real_operation_costs('remove', self.emb_remove,
                                                None, self.num_removes)
        costs_update = sum_real_operation_costs(
            'update', self.emb_update1, self.emb_update2, self.num_updates)

        # the cost regularizer penalizes the weights when the cost per operation
        # goes below 1
        penalty_insert = tf.maximum(
            tf.cast(self.num_inserts, tf.float32) - costs_insert, 0.)
        penalty_remove = tf.maximum(
            tf.cast(self.num_removes, tf.float32) - costs_remove, 0.)
        penalty_update = tf.maximum(
            tf.cast(self.num_updates, tf.float32) - costs_update, 0.)

        penalty_sum = penalty_insert + penalty_remove + penalty_update
        self.transformation_cost_loss = tf.reduce_mean(penalty_sum) * \
            self.cost_regularizer

        self.transformation_cost = costs_insert + costs_remove + costs_update

    def _define_computation_graph(self, create_optimizer):
        """
        Define the tensors for the modeling the TEDIN operations.

        It creates tensors for both the supervised classification step and the
        unsupervised ranking.
        """
        # node arguments of operations; shape is always (batch, max_num_ops, 2)
        # each batch item has all the nodes (word and label indices) given as
        # argument for that operation in that pair
        self.args_insert = tf.placeholder(tf.int32, [None, None, 2],
                                          'args_insert')
        self.args_remove = tf.placeholder(tf.int32, [None, None, 2],
                                          'args_remove')
        self.args1_update = tf.placeholder(tf.int32, [None, None, 2],
                                           'args1_update')
        self.args2_update = tf.placeholder(tf.int32, [None, None, 2],
                                           'args2_update')

        # these are the lengths of the sequences of operations
        self.num_inserts = tf.placeholder(tf.int32, [None], 'num_inserts')
        self.num_removes = tf.placeholder(tf.int32, [None], 'num_removes')
        self.num_updates = tf.placeholder(tf.int32, [None], 'num_updates')

        # these embedded values are (batch, max_ops, embedding_size)
        self.emb_insert = self._embed_nodes(self.args_insert)
        self.emb_remove = self._embed_nodes(self.args_remove)
        self.emb_update1 = self._embed_nodes(self.args1_update)
        self.emb_update2 = self._embed_nodes(self.args2_update)

        # these functions define the rest of the tensors specific for each task
        self._define_transformation_cost()
        self._define_supervised_classifier(create_optimizer)

    def _define_supervised_classifier(self, create_optimizer):
        """
        Define the tensors related to the supervised classification of pairs
        """
        # all representations are (batch, max_ops, hidden_units)
        rep_insert = self.get_operation_representation(
            'insert', self.emb_insert)
        rep_remove = self.get_operation_representation(
            'remove', self.emb_remove)
        rep_update = self.get_operation_representation(
            'update', self.emb_update1, self.emb_update2)

        # run separate convolutions on the 3 sequences and then take one single
        # maximum. this is because joining them could be harder.
        conv_insert = self._convolution(rep_insert, self.num_inserts)
        conv_remove = self._convolution(rep_remove, self.num_removes, True)
        conv_update = self._convolution(rep_update, self.num_updates, True)

        with tf.variable_scope('convolution'):
            tensor_list = [conv_insert, conv_remove, conv_update]
            max_values = tf.reduce_max(tensor_list, 0)
            conv_output = tf.nn.dropout(max_values, self.dropout_keep)

        with tf.variable_scope('softmax', reuse=self.reuse_weights):
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

        if create_optimizer:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
        reuse = reuse or self.reuse_weights
        with tf.variable_scope('convolution', reuse=reuse):
            hidden = tf.layers.dense(inputs, self.num_hidden_units, tf.nn.relu)

            # mask positions after sequence end with -inf
            masked = mask_3d(hidden, lengths, -np.inf)
            max_values = tf.reduce_max(masked, 1)

        return max_values

    def _define_operation_costs(self):
        """
        Define the tensors related to the cost of tree edit operations
        """
        # nodes are the tokens and their dependency relation
        # shape is (batch, sentence_length, 2); 2 means word idx, relation idx
        self.nodes1 = tf.placeholder(tf.int32, [None, None, 2], 'nodes1')
        self.nodes2 = tf.placeholder(tf.int32, [None, None, 2], 'nodes2')

        embedded1 = self._embed_nodes(self.nodes1)
        embedded2 = self._embed_nodes(self.nodes2)
        self.remove_costs = self._define_operation_cost('remove', embedded1)
        self.insert_costs = self._define_operation_cost('insert', embedded2)
        self._define_update_costs(embedded1, embedded2)

    def _define_update_costs(self, embedded1, embedded2):
        """
        Define the tensors for computing update costs.

        It computes all combinations of nodes in the first sentence with the
        second one.

        :param embedded1: tensor (batch, num_nodes1, embedded_size)
        :param embedded2: tensor (batch, num_nodes2, embedded_size)
        """
        num_nodes1 = tf.shape(embedded1)[1]
        num_nodes2 = tf.shape(embedded2)[1]

        # we replicate embedded1 as many times as num_nodes2, and vice-versa
        # both will have shape (batch, num_nodes1, num_nodes2, embedded_size)
        nodes1_4d = tf.expand_dims(embedded1, 2)
        nodes2_4d = tf.expand_dims(embedded2, 1)
        tiled1 = tf.tile(nodes1_4d, [1, 1, num_nodes2, 1])
        tiled2 = tf.tile(nodes2_4d, [1, num_nodes1, 1, 1])
        self.update_costs = self._define_operation_cost('update', tiled1,
                                                        tiled2)

    def _embed_nodes(self, nodes):
        """
        Return the embedded node representation

        :param nodes: either a 2d or 3d tensor (batch, 2) or (batch, num_nodes,
            2). 2 means the word index and node label index.
        :return: either a 2d or 3d tensor
        """
        if len(nodes.get_shape()) == 2:
            word_inds = nodes[:, 0]
            label_inds = nodes[:, 1]
        else:
            word_inds = nodes[:, :, 0]
            label_inds = nodes[:, :, 1]

        embedded_token = tf.nn.embedding_lookup(self.embeddings, word_inds)
        embedded_label = tf.nn.embedding_lookup(self.label_embeddings,
                                                label_inds)
        embedded = tf.concat([embedded_token, embedded_label], -1,
                             'node_embedding')
        return embedded

    def _define_operation_cost(self, scope, node1, node2=None):
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
        reuse = self.reuse_weights or (scope in self._initialized_weights)
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
            # in case of update
            diff = node1 - node2
            inputs = tf.concat([node1, node2, diff], axis=-1)

        reuse = self.reuse_weights or (operation in self._initialized_weights)
        self._initialized_weights.add(operation)
        with tf.variable_scope(operation, reuse=reuse):
            hidden = tf.layers.dense(inputs, self.num_hidden_units, tf.nn.relu)
            hidden = tf.nn.dropout(hidden, self.dropout_keep)

        return hidden

    def run_zss(self, batch, return_costs=False, return_operations=True):
        """
        Run the zhang-shasha algorithm with TEDIN's weights.

        :param batch: Dataset
        :param return_costs: if True, each item in the returned list will
            include the total transformation cost
        :param return_operations: if True, each item in the returned list will
            include the operation list
        :return: if return_costs and return_operations are True, a list of tuples
            (cost, transformations); otherwise or a list of transformations or
            costs. If both are False an exception is thrown.
            Each transformation is a list of zss.Operation
        """
        if not return_costs and not return_operations:
            raise ValueError('It was not specified what to return')

        # precompute the insert and remove costs
        feeds = {self.nodes1: batch.nodes1, self.nodes2: batch.nodes2}

        cost_ops = [self.insert_costs, self.remove_costs, self.update_costs]
        costs = self.session.run(cost_ops, feeds)
        insert_costs, remove_costs, update_costs = costs

        all_operations = []
        for i, item in enumerate(batch):
            cost, operations = find_zss_operations(
                item.pairs, insert_costs[i], remove_costs[i], update_costs[i])

            if return_costs and return_operations:
                all_operations.append((cost, operations))
            elif return_operations:
                all_operations.append(operations)
            else:
                all_operations.append(cost)

        return all_operations

    def create_feeds_from_operations(self, operations):
        """
        Create a feed dictionary from a list of operations

        :param operations: list of lists of operations (batch, num_operations)
        :return: dictionary
        """
        all_inserts = []
        all_removes = []
        all_updates1 = []
        all_updates2 = []

        for op_list in operations:
            inserts = []
            removes = []
            updates1 = []
            updates2 = []

            for op in op_list:
                a1 = op.arg1
                a2 = op.arg2
                if op.type == INSERT:
                    inserts.append([a2.index, a2.dep_index])
                elif op.type == REMOVE:
                    removes.append([a1.index, a1.dep_index])
                elif op.type == UPDATE or op.type == MATCH:
                    updates1.append([a1.index, a1.dep_index])
                    updates2.append([a2.index, a2.dep_index])

            all_inserts.append(inserts)
            all_removes.append(removes)
            all_updates1.append(updates1)
            all_updates2.append(updates2)

        insert_args, num_inserts = utils.nested_list_to_array(
            all_inserts, dim3=2)
        remove_args, num_removes = utils.nested_list_to_array(
            all_removes, dim3=2)
        update_args1, num_updates = utils.nested_list_to_array(
            all_updates1, dim3=2)
        update_args2, _ = utils.nested_list_to_array(
            all_updates2, dim3=2)

        feeds = {self.args_insert: insert_args, self.num_inserts: num_inserts,
                 self.args_remove: remove_args, self.num_removes: num_removes,
                 self.args1_update: update_args1, self.num_updates: num_updates,
                 self.args2_update: update_args2}

        return feeds

    def _create_data_feeds(self, batch):
        """
        Run the zhang-shasha algorithm to find the sequence of edit operations.

        :param batch: Dataset
        :return: dictionary of feeds with the operations and their arguments
        """
        operations = self.run_zss(batch)
        feeds = self.create_feeds_from_operations(operations)
        feeds[self.labels] = batch.labels
        return feeds

    def initialize(self, embeddings):
        """
        Initialize trainable variables and embeddings.

        :param embeddings: numpy array matching the shape given to the
            constructor.
        """
        self.logger.debug('Initializing variables')
        feed = {self.embedding_ph: embeddings}
        self.session.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()], feed)

    def _get_next_batch(self, data, batch_size, training):
        return data.next_batch(batch_size, wrap=training)

    @property
    def train_fetches(self):
        return [self.loss, self.update_acc_op, self.train_op]

    @property
    def validation_fetches(self):
        return [self.accuracy, self.loss]

    def _reset_metrics(self):
        """
        Reset performance metrics
        """
        metric_vars = [v for v in tf.local_variables()
                       if 'accuracy' in v.name]
        self.session.run(tf.variables_initializer(metric_vars))

    def _init_train_stats(self, params, report_interval):
        self._best_acc = 0
        self._accumulated_training_loss = 0
        self._loss_denominator = report_interval

    def _init_validation(self, data):
        self._accumulated_validation_loss = 0
        self._accumulated_validation_acc = 0
        data.reset_batch_counter()

    def _update_training_stats(self, values, data):
        loss, _, _ = values
        self._accumulated_training_loss += loss

    def _update_validation_stats(self, values, data):
        acc, loss = values
        self._accumulated_validation_loss += loss * len(data)
        self._accumulated_validation_acc += acc * len(data)

    def _get_validation_metrics(self, data):
        acc = self._accumulated_validation_acc / len(data)
        loss = self._accumulated_validation_loss / len(data)
        return acc, loss

    def _validation_report(self, values, saver, step, train_data, model_dir):
        valid_acc, valid_loss = values

        train_acc = self.session.run(self.moving_acc)
        train_loss = self._accumulated_training_loss / self._loss_denominator
        self._accumulated_training_loss = 0
        self._reset_metrics()

        msg = '{} epochs\t{} steps\tTrain acc: {:.5}\t' \
              'Train loss: {:.5}\tValid acc: {:.5}\tValid loss: {:.5}'

        if valid_acc > self._best_acc:
            saver.save(self.session, self.path)
            self._best_acc = valid_acc
            msg += ' (saved model)'

        print(msg.format(train_data.epoch, step, train_acc,
                                    train_loss, valid_acc, valid_loss))
