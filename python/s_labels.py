"""Convert characters (chr) to integer (int) labels and vice versa.

`ctc_loss`_ maps labels from 0=<space>, 1=a, ..., 26=z, 27=<blank>

See: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
"""


__map = r'% abcdefghijklmnopqrstuvwxyz'      # 27 characters including <space>.
__ctoi = dict()
__itoc = dict()

if len(__ctoi) == 0 or len(__itoc) == 0:
    for i, c in enumerate(__map):
        __ctoi.update({c: i})
        __itoc.update({i: c})


def ctoi(char):
    """Convert character label to integer.

    Args:
        char (char): Character label.

    Returns:
        int: Integer representation.
    """
    if char not in __map:
        raise ValueError('Invalid input character \'{}\'.'.format(char))
    if not len(char) == 1:
        raise ValueError('"{}" is not a valid character.'.format(char))

    return __ctoi[char.lower()]


def itoc(integer):
    """Convert integer label to character.

    Args:
        integer (int): Integer label.

    Returns:
        char: Character representation.
    """
    if not 0 <= integer < num_classes():
        raise ValueError('Integer label ({}) out of range.'.format(integer))

    return __itoc[integer]


def num_classes():
    """Return number of different classes.

    Returns:
        int: Number of classes.
    """
    return len(__map) + 1
