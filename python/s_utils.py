"""Provides utility functions for the speech recognition network."""

import numpy as np
import tensorflow as tf


class LabelManager(object):
    """Convert characters (chr) to integer (int) labels and vice versa."""
    # TODO: Map from 0=<space>,  1=a, ... to 26=z, 27=<blank>
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    # TODO: <blank label>, <space>

    def __init__(self):
        self._map = ' abcdefghijklmnopqrstuvwxyz '
        self._ctoi = dict()
        self._itoc = dict()

        for i, c in enumerate(self._map):
            self._ctoi.update({c: i})
            self._itoc.update({i: c})

    def ctoi(self, char):
        # L8ER Documentation
        # review: No if `char` exists validation
        if not len(char) == 1:
            raise ValueError('"{}" is not a valid character.'.format(char))
        return self._ctoi[char.lower()]

    def itoc(self, integer):
        # L8ER Documentation
        # review: No if `integer` exists validation
        return self._itoc[integer]

    def num_classes(self):
        """Return number of different classes.

        Returns:
            int: Number of classes.
        """
        return len(self._map)


######################################################################
# CTC Helper
######################################################################
def convert_to_sparse_tensor(labels):
    # TODO Documentation
    indices = list()
    values = list()

    for i, label in enumerate(labels):
        indices.extend((zip([i] * len(label), range(len(label)))))
        values.extend(label)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

    return indices, values, shape


def dense_tensor_to_sparse(dense):
    """Takes in a dense Tensor and returns a SparseTensor.

    # review Unused?

    Args:
        dense (tf.Tensor): A dense Tensor.

    Returns:
        tf.SparseTensor:
            Spare representation of the dense Tensor.
    """
    print('DEBUG:', dense.shape, tf.shape(dense, out_type=tf.int32))
    print('DEBUG2:', dense)
    dense_t = tf.constant(dense)
    idx = tf.where(tf.not_equal(dense_t, 0))
    dense_shape = tf.shape(dense_t, out_type=tf.int32)
    sparse_t = tf.SparseTensor(idx, tf.gather_nd(dense_t, idx), dense_shape)

    assert np.all(dense == tf.sparse_tensor_to_dense(sparse_t))     # L8ER validation, remove this
    return sparse_t
