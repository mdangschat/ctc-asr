"""Provides utility functions for the speech recognition network."""


class LabelManager(object):
    """Convert characters (chr) to integer (int) labels and vice versa."""
    # L8ER: <blank label>, <space>
    # TODO: Map from 1 to x, not from 0 to x-1

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
