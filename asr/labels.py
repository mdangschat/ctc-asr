"""
Convert characters (chr) to integer (int) labels and vice versa.

REVIEW: index 0 bug, also see:
https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding

`ctc_loss`_ maps labels from 0=<unused>, 1=<space>, 2=a, ..., 27=z, 28=<blank>

See: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
"""

__MAP = r' abcdefghijklmnopqrstuvwxyz'  # 27 characters including <space>.
__CTOI = dict()
__ITOC = dict([(0, '')])  # This is in case the net decodes a 0 on step 0.

if not __CTOI or not __ITOC:
    for i, c in enumerate(__MAP):
        __CTOI.update({c: i + 1})
        __ITOC.update({i + 1: c})


def ctoi(char):
    """
    Convert character label to integer.

    Args:
        char (char): Character label.

    Returns:
        int: Integer representation.
    """
    if char not in __MAP:
        raise ValueError('Invalid input character \'{}\'.'.format(char))
    if not len(char) == 1:
        raise ValueError('"{}" is not a valid character.'.format(char))

    return __CTOI[char.lower()]


def itoc(integer):
    """
    Convert integer label to character.

    Args:
        integer (int): Integer label.

    Returns:
        char: Character representation.
    """
    if not 0 <= integer < num_classes():
        raise ValueError('Integer label ({}) out of range.'.format(integer))

    return __ITOC[integer]


def num_classes():
    """
    Return number of different classes, +1 for the <blank> label.

    Returns:
        int: Number of labels +1.
    """
    return len(__MAP) + 2
