# -*- coding: utf-8 -*-

"""
Class for a model that learns weights for different tree edit operations.

The treee edit distance code was adapted from the Python module zss.
"""

import tensorflow as tf

from src.trainable import Trainable
from src.datastructures import Dataset


# operation codes
INSERT = 1
REMOVE = 2
UPDATE = 3
MATCH = 4


def index_columns(tensor, indices):
    """
    Return the elements of a tensor such that its n-th row is indexed by the
    n-th index in the given indices.

    :param tensor: 2d tensor
    :param indices: 1d tensor
    """
    batch_size = tf.shape(indices)[0]
    indices_2d = tf.stack([tf.range(batch_size), indices], axis=1)
    return tf.gather_nd(tensor, indices_2d)


def append_operations(operations1, operations2):
    """
    Append all the operations2 to operations1

    0 indicates padding

    :param operations1: a 2d tensor (batch, max_num_ops)
    :param operations2: a 2d tensor (batch, max_num_ops)
    :return:
    """


def row_wise_gather(tensor, indices, clip_inds=True):
    """
    Apply the equivalent of tf.gather(tensor[i], indices[i]) for every row i
    in `tensor` and `indices`.

    :param tensor: 2d tensor
    :param indices: 2d tensor
    :param clip_inds: if True, any value in indices past the end of dim 1 of
        tensor are clipped to the maximum allowed.
    :return: 2d tensor
    """
    if clip_inds:
        max_dim2 = tf.shape(tensor)[1]
        clip_value = (max_dim2 - 1) * tf.ones_like(indices)
        indices = tf.where(indices > clip_value, clip_value, indices)

    # tensor = tf.Print(tensor, [tf.shape(tensor),
    #                            tf.shape(indices), indices],
    #                   'tensor, indices', summarize=30)
    gathered3d = tf.gather(tensor, indices, axis=1)

    # we must take the vectors at position [0, 0], [1, 1], [2, 2] and so on
    batch_size = tf.shape(gathered3d)[0]
    range_ = tf.range(batch_size)
    inds = tf.tile(tf.reshape(range_, [-1, 1]), [1, 2])

    return tf.gather_nd(gathered3d, inds)


def index_3d(tensor, inds1, inds2):
    """
    Return the elements of a tensor [i, inds1[i], inds2[i]], with
    0 <= i <= num_rows

    :param tensor: 3d tensor
    :param inds1: 1d tensor
    :param inds2: 1d tensor
    :return: 1d tensor
    """
    batch_size = tf.shape(inds1)[0]
    inds = tf.stack([tf.range(batch_size), inds1, inds2], axis=1)
    return tf.gather_nd(tensor, inds)


class TreeEditDistanceNetwork(Trainable):
    """
    Model that learns weights for different tree edit operations.
    """

    def __init__(self, vocabulary_size, embedding_size):
        super(TreeEditDistanceNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocabulary_size

        # each sentence pair is represented by the nodes (words), lmd (left-most
        # descendants), and keyroots of each sentence

        # nodes are the tokens themselves (word indices)
        self.nodes1 = tf.placeholder(tf.int32, [None, None], 'nodes1')
        self.nodes2 = tf.placeholder(tf.int32, [None, None], 'nodes2')

        # leftmost descendents
        self.lmd1 = tf.placeholder(tf.int32, [None, None],
                                   'leftmost-descendants1')
        self.lmd2 = tf.placeholder(tf.int32, [None, None],
                                   'leftmost-descendants2')

        # key roots
        self.key_roots1 = tf.placeholder(tf.int32, [None, None], 'keyroots1')
        self.key_roots2 = tf.placeholder(tf.int32, [None, None], 'keyroots2')

        # sentence sizes
        self.sizes1 = tf.placeholder(tf.int32, [None], 'sizes1')
        self.sizes2 = tf.placeholder(tf.int32, [None], 'sizes2')

        # number of actual key roots
        self.num_keyroots1 = tf.placeholder(tf.int32, [None], 'num_key_roots1')
        self.num_keyroots2 = tf.placeholder(tf.int32, [None], 'num_key_roots2')

        # target task label
        self.label = tf.placeholder(tf.int32, [None], 'label')

        # training params
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')

        # remove_costs stores the costs for removing each node1
        # insert_costs stores the costs for inserting each node2
        # both shapes are (batch, num_units)
        self.remove_costs = self.compute_remove_cost(self.nodes1)
        self.insert_costs = self.compute_insert_cost(self.nodes2)

        self.tree_distance = self.compute_distance()
        self.init_op = self._init_internal_variables()

    def _init_internal_variables(self):
        return tf.variables_initializer([self.distances, self.fd,
                                         self.tree_ops, self.fd_ops,
                                         self.backpointer_col,
                                         self.backpointer_row])

    def _create_batch_feed(self, batch, **kwargs):
        """
        Create a feed dictionary for the placeholders.

        :type batch: Dataset
        :param kwargs:
        :return:
        """
        feeds = {self.sizes1: batch.sizes1, self.sizes2: batch.sizes2,
                 self.nodes1: batch.nodes1, self.nodes2: batch.nodes2,
                 self.lmd1: batch.lmd1, self.lmd2: batch.lmd2,
                 self.key_roots1: batch.key_roots1,
                 self.key_roots2: batch.key_roots2,
                 self.num_keyroots1: batch.num_key_roots1,
                 self.num_keyroots2: batch.num_key_roots2}

        return feeds

    def compute_remove_cost(self, nodes):
        """
        :param nodes: tensor (batch, num_nodes)
        :param nodes: tensor (batch,)
        :return: tensor (batch, num_nodes)
        """
        costs = tf.ones_like(nodes, tf.float32)
        # ranges = tf.range(tf.reduce_max(sizes))
        # tf.tile(tf.reshape(ranges, [-1, 1]),

        return costs

    def compute_insert_cost(self, nodes):
        """
        :param nodes: tensor (batch, num_nodes)
        :return: tensor (batch, num_nodes)
        """
        return tf.ones_like(nodes, tf.float32)

    def compute_update_cost(self, node1, node2):
        """
        :param node1: tensor (batch, num_units)
        :param node2: tensor (batch, num_units)
        :return: tensor (batch)
        """
        return 1 - tf.cast(tf.equal(node1, node2), tf.float32)

    def _assign_inital_insert_operations(self, ops):
        """
        Create an operator to assign values to a variable such that the i-th
        column has a sequence of i INSERT operations.

        :param ops: tensor to hold the op codes, shape (batch, num_nodes1,
            num_nodes2, num_nodes1 + num_nodes2)
        :return: an assign operator
        """
        r = tf.range(tf.shape(ops)[-1])
        tiled_r = tf.tile(tf.reshape(r, [1, -1]), [tf.shape(ops)[2], 1])
        num_ops = tf.reshape(tf.range(tf.shape(ops)[2]), [-1, 1])
        new_ops = INSERT * tf.cast(tf.less(tiled_r, num_ops), tf.int32)
        new_ops3d = tf.reshape(new_ops, [1, tf.shape(ops)[2],
                                         tf.shape(ops)[-1]])
        tiled_new_ops = tf.tile(new_ops3d, [tf.shape(ops)[0], 1, 1])

        return tf.assign(ops[:, 0, :, :], tiled_new_ops)

    def _assign_inital_remove_operations(self, ops):
        """
        Create an operator to assign values to a variable such that the i-th
        row has a sequence of i REMOVE operations.

        :param ops: tensor to hold the op codes, shape (batch, num_nodes1,
            num_nodes2, num_nodes1 + num_nodes2)
        :return: an assign operator
        """
        r = tf.range(tf.shape(ops)[-1])
        tiled_r = tf.tile(tf.reshape(r, [1, -1]), [tf.shape(ops)[1], 1])
        num_ops = tf.reshape(tf.range(tf.shape(ops)[1]), [-1, 1])
        new_ops = REMOVE * tf.cast(tf.less(tiled_r, num_ops), tf.int32)
        new_ops3d = tf.reshape(new_ops, [1, tf.shape(ops)[1],
                                         tf.shape(ops)[-1]])
        tiled_new_ops = tf.tile(new_ops3d, [tf.shape(ops)[0], 1, 1])

        return tf.assign(ops[:, :, 0, :], tiled_new_ops)

    def compute_distance(self):
        """
        Compute the distance between the two given sentences in placeholders

        :return: the computed minimum tree edit distance
        """
        batch_size = tf.shape(self.sizes1)[0]
        max_size1 = tf.reduce_max(self.sizes1)
        max_size2 = tf.reduce_max(self.sizes2)

        # these are the distances between each pair of trees
        distances = tf.Variable(
            tf.zeros([batch_size, max_size1, max_size2]), trainable=False,
            validate_shape=False, name='distances')

        # fd stores partial distances between subtrees; it is overwritten inside
        # loops
        fd_values = tf.zeros((batch_size, max_size1 + 1, max_size2 + 1),
                             tf.float32)
        fd = tf.Variable(fd_values, trainable=False, validate_shape=False,
                         name='forest_distances')

        self.fd = fd
        self.distances = distances

        # this stores the operations used in each subtree cell
        # each cell holds the operation value; the nodes involved are determined
        # by its coordinates
        # partial_ops = tf.Variable(tf.zeros_like(fd_values, tf.int32),
        #                           trainable=False, validate_shape=False,
        #                           name='partial_operations')
        shape = [batch_size, max_size1 + 1, max_size2 + 1,
                 max_size1 + max_size2]
        fd_ops = tf.Variable(tf.zeros(shape, tf.int32), trainable=False,
                             validate_shape=False, name='fd_operations')

        # this stores the list of operations in each tree distance cell
        shape = [batch_size, max_size1, max_size2, max_size1 + max_size2]
        tree_ops = tf.Variable(tf.zeros(shape, tf.int32), trainable=False,
                                 validate_shape=False, name='operations')
        # self.partial_ops = partial_ops
        self.fd_ops = fd_ops
        self.tree_ops = tree_ops

        # [backpointer_row[i, j], backpointer_col[i, j]]  points to the last
        # operation performed before operations[i, j]
        bp_row = tf.Variable(
            tf.zeros_like(fd_values, tf.int32), trainable=False,
            validate_shape=False, name='backpointer_row')
        bp_col = tf.Variable(
            tf.zeros_like(fd_values, tf.int32), trainable=False,
            validate_shape=False, name='backpointer_col')
        self.backpointer_row = bp_row
        self.backpointer_col = bp_col

        def outer_keyroot_loop(i):
            # loops along keyroots of sentence 1
            max_j = tf.reduce_max(self.num_keyroots2)
            cond = lambda i, j: j < max_j
            loop = tf.while_loop(cond, inner_keyroot_loop, [i, 0])
            with tf.control_dependencies(loop):
                i += 1

            return i

        def inner_keyroot_loop(i_idx, j_idx):
            """
            Loop along keyroots of sentence 2
            computes the tree distance
            """
            i = self.key_roots1[:, i_idx]
            j = self.key_roots2[:, j_idx]

            # m and n have shape (batch_size,)
            lmd1_i = index_columns(self.lmd1, i)
            m = i - lmd1_i + 2

            lmd2_j = index_columns(self.lmd2, j)
            n = j - lmd2_j + 2

            max_m = tf.reduce_max(m)
            max_n = tf.reduce_max(n)

            # reset the fd and partial_ops values
            assign_fd = tf.assign(fd, tf.zeros_like(fd, tf.float32))
            assign_ops = tf.assign(fd_ops, tf.zeros_like(fd_ops))
            # assign_ops = tf.assign(partial_ops, tf.zeros_like(partial_ops))

            with tf.control_dependencies([assign_fd, assign_ops]):
                # ioff and joff have shape (batch_size,)
                ioff = lmd1_i - 1
                joff = lmd2_j - 1

            # here we set the all-remove and all-insert operations
            ranges = tf.tile(tf.reshape(tf.range(1, max_m), [1, -1]),
                             [batch_size, 1])
            inds = ranges + tf.reshape(ioff, [-1, 1])
            remove_costs = row_wise_gather(self.remove_costs, inds)
            cumulative_costs = tf.cumsum(remove_costs, axis=1)
            assign_remove = tf.assign(fd[:, 1:max_m, 0], cumulative_costs)
            # remove_op = REMOVE * tf.ones_like(partial_ops[:, 1:max_m, 0])
            # assign_remove_ops = tf.assign(partial_ops[:, 1:max_m, 0], remove_op)
            assign_remove_ops = self._assign_inital_remove_operations(fd_ops)
            # assign_row = tf.assign(bp_row[:, :, 0], tf.range(max_m))

            ranges = tf.tile(tf.reshape(tf.range(1, max_n), [1, -1]),
                             [batch_size, 1])
            inds = ranges + tf.reshape(joff, [-1, 1])

            insert_costs = row_wise_gather(self.insert_costs, inds)
            cumulative_costs = tf.cumsum(insert_costs, axis=1)
            # insert_op = INSERT * tf.ones_like(partial_ops[:, 0, 1:max_n])
            assign_insert = tf.assign(fd[:, 0, 1:max_n], cumulative_costs)
            # assign_insert_ops = tf.assign(partial_ops[:, 0, 1:max_n], insert_op)
            assign_insert_ops = self._assign_inital_insert_operations(fd_ops)
            # assign_col = tf.assign(bp_col[:, 0, :], tf.range(max_n))

            # I heard you like nested functions so I put a nested function
            # inside your nested function
            def outer_loop(x):
                loop = tf.while_loop(lambda x, y: y < max_n, inner_loop, [x, 1])
                with tf.control_dependencies(loop):
                    x += 1

                return x

            def inner_loop(x, y):
                lmd1_i = index_columns(self.lmd1, i)

                # x is iterating from 1 to max_m -- which may be higher than
                # the tensor bounds for some items in the batch
                # clip it to the last valid one
                clipped_x = tf.where(x > m, m, x * tf.ones_like(m))
                clipped_y = tf.where(y > n, n, y * tf.ones_like(n))
                lmd1_x_ioff = index_columns(self.lmd1, clipped_x + ioff)
                condition1 = tf.equal(lmd1_i, lmd1_x_ioff)

                lmd2_j = index_columns(self.lmd2, j)
                lmd2_y_joff = index_columns(self.lmd2, clipped_y + joff)
                condition2 = tf.equal(lmd2_j, lmd2_y_joff)

                condition = tf.logical_and(condition1, condition2)

                nodes1 = index_columns(self.nodes1, clipped_x + ioff)
                nodes2 = index_columns(self.nodes2, clipped_y + joff)

                cost1 = index_columns(self.remove_costs, clipped_x + ioff)
                cost2 = index_columns(self.insert_costs, clipped_y + joff)
                cost_remove = fd[:, x - 1, y] + cost1
                cost_insert = fd[:, x, y - 1] + cost2

                cost_update_positive = fd[:, x - 1, y - 1] + \
                    self.compute_update_cost(nodes1, nodes2)

                p = lmd1_x_ioff - 1 - ioff
                q = lmd2_y_joff - 1 - joff

                # avoid the indices x + ioff and y + joff going after the
                # tensor bounds. this happens only when "condition" is false and
                # they aren't needed; we clip them for computational reasons
                old_distance = index_3d(distances, clipped_x + ioff,
                                        clipped_y + joff)
                old_operations = index_3d(tree_ops, clipped_x + ioff,
                                          clipped_y + joff)

                # tf.gather_nd doesn't accept negative indices, so we just
                # replace any of them with 0. This has no influence in the
                # result, because p and q are always positive when "condition"
                # is false
                p = tf.where(p < 0, tf.zeros_like(p), p)
                q = tf.where(q < 0, tf.zeros_like(q), q)
                old_fd = index_3d(fd, p, q)
                cost_update_negative = old_fd + old_distance

                cost_update = tf.where(condition, cost_update_positive,
                                       cost_update_negative)

                costs = [cost_insert, cost_remove, cost_update]
                min_cost = tf.reduce_min(costs, axis=0)

                # codes are insert = 1 , remove = 2, update = 3
                # 0 should only appear as padding
                operation = tf.cast(tf.argmin(costs, axis=0) + 1, tf.int32)

                assign_fd = tf.assign(fd[:, x, y], min_cost)
                # assign_ops = tf.assign(partial_ops[:, x, y], operation)

                # each of the history tensors has the history of operations
                # used up to the previous step. Shape is (batch, max_ops)
                history_insert = fd_ops[:, x, y - 1, :]
                history_remove = fd_ops[:, x - 1, y, :]
                history_update = tf.where(condition, fd_ops[:, x - 1, y - 1, :],
                                          index_3d(fd_ops, p, q))

                # op_hist is the history of operations before the one used in
                # this time step
                # operation has shape (batch,); tf.where broadcasts it
                op_hist = tf.where(
                    tf.equal(operation, INSERT), history_insert,
                    tf.where(tf.equal(operation, REMOVE), history_remove,
                             history_update))

                # assign the new operation in the first position
                assign_ops = tf.assign(fd_ops[:, x, y, 0], operation)
                assign_op_hist = tf.assign(fd_ops[:, x, y, 1:], op_hist[:, :-1])

                # when you go down a row, you remove a node
                # when you go right a column, you add a node
                ones = tf.ones([batch_size], tf.int32)
                rows = tf.where(tf.equal(min_cost, INSERT),
                                x * ones, (x - 1) * ones)
                cols = tf.where(tf.equal(min_cost, REMOVE),
                                y * ones, (y - 1) * ones)
                assign_rows = tf.assign(bp_row[:, x, y], rows)
                assign_cols = tf.assign(bp_col[:, x, y], cols)
                deps = [assign_fd, assign_ops, assign_rows, assign_cols,
                        assign_op_hist]
                with tf.control_dependencies(deps):
                    # in practice, we want to do something like
                    # if condition:
                    #     distances[:, x + ioff, y + joff] = new_distance
                    # condition is (batch_size,)
                    new_distance_values = tf.where(condition, min_cost,
                                                   old_distance)
                    new_operation_values = tf.where(condition, fd_ops[:, x, y],
                                                    old_operations)

                inds = tf.stack([tf.range(batch_size), clipped_x + ioff,
                                 clipped_y + joff],
                                axis=1)
                assign_dists = tf.scatter_nd_update(distances, inds,
                                                    new_distance_values)
                assign_ops = tf.scatter_nd_update(tree_ops, inds,
                                                  new_operation_values)
                with tf.control_dependencies([assign_dists, assign_ops]):
                    y += 1

                return x, y

            # this is the main nested loop for the rest of the comparisons
            dependencies = [assign_remove, assign_insert,
                            assign_remove_ops, assign_insert_ops,
                            # assign_row, assign_col
                            ]
            with tf.control_dependencies(dependencies):
                loop3 = tf.while_loop(lambda x: x < max_m, outer_loop, [1])

            with tf.control_dependencies([loop3]):
                j_idx += 1

            return i_idx, j_idx

        max_keyroots1 = tf.reduce_max(self.num_keyroots1)
        cond = lambda i: i < max_keyroots1
        loop = tf.while_loop(cond, outer_keyroot_loop, [0])

        with tf.control_dependencies([loop]):
            # create a 2d tensor where each row (i, j, k) has indicates the last
            # position in the i-th pair (last position j of sentence 1 and k of
            # sentence 2)
            inds = tf.stack([tf.range(batch_size), self.sizes1 - 1,
                             self.sizes2 - 1], axis=1)
            last_distance = tf.gather_nd(distances, inds)

        return last_distance
