"""Provides utility functions for the speech recognition network."""

import numpy as np


class LabelManager(object):
    """Convert characters (chr) to integer (int) labels and vice versa."""
    # TODO: Map from 0=<space>,  1=a, ... to 26=z, 27=<blank>
    # review: <blank label>, <space>
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss

    def __init__(self):
        self._map = r' abcdefghijklmnopqrstuvwxyz'      # 27 characters including <space>.
        self._ctoi = dict()
        self._itoc = dict()

        for i, c in enumerate(self._map):
            self._ctoi.update({c: i})
            self._itoc.update({i: c})

    def ctoi(self, char):
        """Convert character label to integer.

        Args:
            char (char): Character label.

        Returns:
            int:
                Integer representation.
        """
        if char not in self._map:
            raise ValueError('Invalid input character \'{}\'.'.format(char))
        if not len(char) == 1:
            raise ValueError('"{}" is not a valid character.'.format(char))

        return self._ctoi[char.lower()]

    def itoc(self, integer):
        """Convert integer label to character.

        Args:
            integer (int): Integer label.

        Returns:
            char:
                Character representation.
        """
        if not 0 <= integer < self.num_classes():
            raise ValueError('Integer label ({}) out of range.'.format(integer))

        return self._itoc[integer]

    def num_classes(self):
        """Return number of different classes.

        Returns:
            int:
                Number of classes.
        """
        return len(self._map) + 1


######################################################################
# CTC Helper
######################################################################
def convert_to_sparse_tensor(labels):
    # L8ER: Documentation
    # Review: Method not used at the moment.
    indices = list()
    values = list()

    for i, label in enumerate(labels):
        indices.extend((zip([i] * len(label), range(len(label)))))
        values.extend(label)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

    return indices, values, shape
