"""Convert characters (chr) to integer (int) labels and vice versa.

REVIEW: index 0 bug, also see:
<https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding>

`ctc_loss`_ maps labels from 0=<unused>, 1=<space>, 2=a, ..., 27=z, 28=<blank>

See: https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
"""


__map = r' abcdefghijklmnopqrstuvwxyz'      # 27 characters including <space>.
__ctoi = dict()
__itoc = dict([(0, '')])   # This is in case the net decodes a 0 on step 0.

if len(__ctoi) == 0 or len(__itoc) == 0:
    for i, c in enumerate(__map):
        __ctoi.update({c: i + 1})
        __itoc.update({i + 1: c})


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
    """Return number of different classes +1 for the <blank> label.

    Returns:
        int: Number of labels +1.
    """
    return len(__map) + 2
