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


def index_3d(tensor, inds1, inds2):
    """
    Return the elements of a tensor [i, inds1[i], inds2[i]], with i = number of
    items / rows.

    :param tensor: 2d tensor
    :param inds1: 1d tensor
    :param inds2: 1d tensor
    :return: 1d tensor
    """
    batch_size = tf.shape(inds1)[0]
    inds = tf.stack([tf.range(batch_size), inds1, inds2], axis=1)
    return tf.gather_nd(tensor, inds)


class SimpleTreeEditLearner(Trainable):
    """
    Model that learns weights for different tree edit operations.
    """

    def __init__(self, vocabulary_size, embedding_size):
        super(Trainable, self).__init__()

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
        self.keyroots1 = tf.placeholder(tf.int32, [None, None], 'keyroots1')
        self.keyroots2 = tf.placeholder(tf.int32, [None, None], 'keyroots2')

        # sentence sizes
        self.sizes1 = tf.placeholder(tf.int32, [None], 'sizes1')
        self.sizes2 = tf.placeholder(tf.int32, [None], 'sizes2')

        # number of actual key roots
        self.num_keyroots1 = tf.placeholder(tf.int32, [None], 'num_keyroots1')
        self.num_keyroots2 = tf.placeholder(tf.int32, [None], 'num_keyroots2')

        # target task label
        self.label = tf.placeholder(tf.int32, [None], 'label')

        # training params
        self.learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')

        # self.tree_distance = self.compute_distance()

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
                 self.keyroots1: batch.key_roots1,
                 self.keyroots2: batch.key_roots2,
                 self.num_keyroots1: batch.num_key_roots1,
                 self.num_keyroots2: batch.num_key_roots2}

        return feeds

    def remove_cost(self, node):
        return 1

    def insert_cost(self, node):
        return 1

    def update_cost(self, node1, node2):
        return 2

    def compute_distance(self):
        """
        Compute the distance between the two given sentences in placeholders

        :return: the computed minimum tree edit distance
        """
        batch_size = tf.shape(self.sizes1)[0]
        max_size1 = tf.reduce_max(self.sizes1)
        max_size2 = tf.reduce_max(self.sizes2)
        with tf.control_dependencies(None):
            # hack to create variable in a function called by a loop
            distances = tf.Variable(
                tf.zeros([batch_size, max_size1, max_size2]), trainable=False,
                validate_shape=False)

        # fd stores partial distances between subtrees; it is overwritten inside
        # loops
        fd_values = tf.zeros((batch_size, max_size1 + 1, max_size2 + 1),
                             tf.float32)
        fd = tf.Variable(fd_values, trainable=False, validate_shape=False)

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
            i = self.keyroots1[:, i_idx]
            j = self.keyroots2[:, j_idx]

            # m and n have shape (batch_size,)
            lmd1_i = index_columns(self.lmd1, i)
            m = i - lmd1_i + 2

            lmd2_j = index_columns(self.lmd2, j)
            n = j - lmd2_j + 2

            max_m = tf.reduce_max(m)
            max_n = tf.reduce_max(n)

            # reset the fd values
            assign = tf.assign(fd, tf.zeros_like(fd, tf.float32))

            # # fd stores partial distances between subtrees
            # with tf.control_dependencies(None):
            #     # hack to create variable in a function called by a loop
            #     fd = tf.Variable(tf.zeros((batch_size, max_m, max_n),
            #                               tf.float32),
            #                      trainable=False, validate_shape=False)

            with tf.control_dependencies([assign]):
                ioff = lmd1_i - 1
                joff = lmd2_j - 1

            # I heard you like nested functions so I put a nested function
            # inside your nested function
            def inner_loop_remove(x):
                node = index_columns(self.nodes1, x + ioff)
                assign = tf.assign(fd[:, x, 0],
                                   fd[:, x - 1, 0] + self.remove_cost(node))
                with tf.control_dependencies([assign]):
                    x += 1

                return x

            def inner_loop_insert(y):
                node = index_columns(self.nodes2, y + joff)
                assign = tf.assign(fd[:, 0, y],
                                   fd[:, 0, y - 1] + self.insert_cost(node))
                with tf.control_dependencies([assign]):
                    y += 1

                return y

            # these two loops set the all-remove and all-insert solution
            x = tf.constant(1)
            loop1 = tf.while_loop(lambda x: x < max_m, inner_loop_remove, [x],
                                  parallel_iterations=1)
            loop2 = tf.while_loop(lambda x: x < max_n, inner_loop_insert, [x],
                                  parallel_iterations=1)

            def outer_loop(x):
                loop = tf.while_loop(lambda x, y: y < max_n, inner_loop, [x, 1])
                with tf.control_dependencies(loop):
                    x += 1

                return x

            def inner_loop(x, y):
                lmd1_i = index_columns(self.lmd1, i)
                lmd1_x_ioff = index_columns(self.lmd1, x + ioff)
                condition1 = tf.equal(lmd1_i, lmd1_x_ioff)

                lmd2_j = index_columns(self.lmd2, j)
                lmd2_y_joff = index_columns(self.lmd2, y + joff)
                condition2 = tf.equal(lmd2_j, lmd2_y_joff)

                condition = tf.logical_and(condition1, condition2)

                nodes1 = index_columns(self.nodes1, x + ioff)
                nodes2 = index_columns(self.nodes2, y + joff)
                cost_insert = fd[:, x - 1, y] + self.remove_cost(nodes1)
                cost_remove = fd[:, x, y - 1] + self.insert_cost(nodes2)

                cost_update_positive = fd[:, x - 1, y - 1] + \
                    self.update_cost(nodes1, nodes2)

                p = lmd1_x_ioff - 1 - ioff
                q = lmd2_y_joff - 1 - joff
                old_distance = index_3d(distances, x + ioff, y + joff)

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

                inds = tf.stack([tf.range(batch_size), x + ioff, y + joff],
                                axis=1)
                assign = tf.scatter_nd_update(distances, inds,
                                              new_distance_values)
                with tf.control_dependencies([assign]):
                    y += 1

                return x, y

            # this is the main nested loop for the rest of the comparisons
            with tf.control_dependencies([loop1, loop2]):
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
            inds = tf.stack([tf.range(batch_size), self.sizes1, self.sizes2],
                            axis=1)
            last_distance = tf.gather_nd(distances, inds)

        return last_distance
