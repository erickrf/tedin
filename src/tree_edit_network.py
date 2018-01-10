# -*- coding: utf-8 -*-

"""
Class for a model that learns weights for different tree edit operations.

The treee edit distance code was adapted from the Python module zss.
"""

import tensorflow as tf

from trainable import Trainable
from datastructures import Dataset


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


class TreeEditNetwork(Trainable):
    """
    Model that learns weights for different tree edit operations.
    """

    def __init__(self, vocabulary_size, embedding_size):
        super(TreeEditNetwork, self).__init__()

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
        return tf.variables_initializer([self.distances, self.fd])

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

    def compute_distance(self):
        """
        Compute the distance between the two given sentences in placeholders

        :return: the computed minimum tree edit distance
        """
        batch_size = tf.shape(self.sizes1)[0]
        max_size1 = tf.reduce_max(self.sizes1)
        max_size2 = tf.reduce_max(self.sizes2)
        distances = tf.Variable(
            tf.zeros([batch_size, max_size1, max_size2]), trainable=False,
            validate_shape=False, name='distances')

        # fd stores partial distances between subtrees; it is overwritten inside
        # loops
        fd_values = tf.zeros((batch_size, max_size1 + 1, max_size2 + 1),
                             tf.float32)
        fd = tf.Variable(fd_values, trainable=False, validate_shape=False,
                         name='partial_distances')

        self.fd = fd
        self.distances = distances

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

            # reset the fd values
            assign = tf.assign(fd, tf.zeros_like(fd, tf.float32))

            with tf.control_dependencies([assign]):
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

            ranges = tf.tile(tf.reshape(tf.range(1, max_n), [1, -1]),
                             [batch_size, 1])
            inds = ranges + tf.reshape(joff, [-1, 1])

            insert_costs = row_wise_gather(self.insert_costs, inds)
            cumulative_costs = tf.cumsum(insert_costs, axis=1)
            assign_insert = tf.assign(fd[:, 0, 1:max_n], cumulative_costs)

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
                # cost_insert = fd[:, x - 1, y] + self.remove_cost(nodes1)
                # cost_remove = fd[:, x, y - 1] + self.insert_cost(nodes2)

                cost1 = index_columns(self.remove_costs, clipped_x + ioff)
                cost2 = index_columns(self.insert_costs, clipped_y + joff)
                cost_insert = fd[:, x - 1, y] + cost1
                cost_remove = fd[:, x, y - 1] + cost2

                cost_update_positive = fd[:, x - 1, y - 1] + \
                    self.compute_update_cost(nodes1, nodes2)

                p = lmd1_x_ioff - 1 - ioff
                q = lmd2_y_joff - 1 - joff

                # avoid the indices x + ioff and y + joff going after the
                # tensor bounds. this happens only when "condition" is false and
                # they aren't needed; we clip them for computational reasons
                old_distance = index_3d(distances, clipped_x + ioff,
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

                min_cost = tf.reduce_min([cost_insert, cost_remove,
                                          cost_update], axis=0)

                assign = tf.assign(fd[:, x, y], min_cost)
                with tf.control_dependencies([assign]):
                    # in practice, we want to do something like
                    # if condition:
                    #     distances[:, x + ioff, y + joff] = new_distance
                    # condition is (batch_size,)
                    new_distance_values = tf.where(condition, min_cost,
                                                   old_distance)

                inds = tf.stack([tf.range(batch_size), clipped_x + ioff,
                                 clipped_y + joff],
                                axis=1)
                assign = tf.scatter_nd_update(distances, inds,
                                              new_distance_values)
                with tf.control_dependencies([assign]):
                    y += 1

                return x, y

            # this is the main nested loop for the rest of the comparisons
            with tf.control_dependencies([assign_remove, assign_insert]):
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
