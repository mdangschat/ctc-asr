"""Provides utility functions for the speech recognition network."""


class LabelManager(object):
    """Convert characters (chr) to integer (int) labels and vice versa."""
    # TODO: Map from 0=a, ... to 25=z, 26=<space>, 27=<blank>
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    # TODO: <blank label>, <space>

    def __init__(self):
        self._map = 'abcdefghijklmnopqrstuvwxyz '
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
            (int) Number of classes.
        """
        return len(self._map)


######################################################################
# CTC Helper
######################################################################
def ctc_sparse_tensor():
    # TODO Documentation
    pass
