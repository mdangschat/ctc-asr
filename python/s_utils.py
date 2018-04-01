"""Provides utility functions for the speech recognition network."""


class LabelManager(object):
    """Convert characters (chr) to integer (int) labels and vice versa.

    Maps from 0=<space>,  1=a, ... to 26=z, 27=<blank>

    See: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    """
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
