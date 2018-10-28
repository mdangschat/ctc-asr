"""Methods the calculate cost metrics."""

import numpy as np

from asr.params import NP_FLOAT
from asr.labels import itoc


def dense_to_text(decoded, originals):
    """Convert a dense, integer encoded `tf.Tensor` into a readable string.
    Create a summary comparing the decoded plaintext with a given original string.

        Args:
            decoded (np.ndarray):
                Integer array, containing the decoded sequences.
            originals (np.ndarray):
                String tensor, containing the original input string for comparision.
                `originals` can be an empty tensor.

        Returns:
            np.ndarray:
                1D string Tensor containing only the decoded text outputs.
                    [decoded_string_0, ..., decoded_string_N]
            np.ndarray:
                2D string Tensor with layout:
                    [[decoded_string_0, original_string_0], ...
                     [decoded_string_N, original_string_N]]
        """
    decoded_strings = []
    original_strings = []

    for d in decoded:
        decoded_strings.append(''.join([itoc(i) for i in d]))

    if len(originals) > 0:
        for o in originals:
            original_strings.append(''.join([c for c in o.decode('utf-8')]))
    else:
        original_strings = ['n/a'] * len(decoded_strings)

    decoded_strings = np.array(decoded_strings, dtype=np.object)
    original_strings = np.array(original_strings, dtype=np.object)

    summary = np.vstack([decoded_strings, original_strings])

    return np.array(decoded_strings), summary


# The following function has been taken from:
# <https://github.com/mozilla/DeepSpeech/blob/master/util/text.py#L85>
def wer(original, result):
    """The Word Error Rate (WER) is defined as the editing/Levenshtein distance
    on word level divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).

    Args:
        original (np.string): The original sentences.
            A tf.Tensor converted to `np.ndarray` object bytes by `tf.py_func`.
        result (np.string): The decoded sentences.
            A tf.Tensor converted to `np.ndarray` object bytes by `tf.py_func`.

    Returns:
        np.ndarray: Numpy array containing float scalar.
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    levenshtein_distance = levenshtein(original, result) / float(len(original))
    return np.array(levenshtein_distance, dtype=NP_FLOAT)


# The following functiom has been taken from:
# <https://github.com/mozilla/DeepSpeech/blob/master/util/text.py#L99>
def wer_batch(originals, results):
    """Calculate the Word Error Rate (WER) for a batch.

    Args:
        originals (np.ndarray): 2D string Tensor with the original sentences. [batch_size, 1]
            A tf.Tensor converted to `np.ndarray` bytes by `tf.py_func`.
        results (np.ndarray): 2D string Tensor with the decoded sentences. [batch_size, 1]
            A tf.Tensor converted to `np.ndarray` bytes by `tf.py_func`.

    Returns:
        np.ndarray:
            Float array containing the WER for every sample within the batch. [batch_size]
        np.ndarray:
            Float scalar with the average WER for the batch.
    """
    count = len(originals)
    rates = np.array([], dtype=NP_FLOAT)
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates = np.append(rates, rate)

    return rates, np.array(mean / float(count), dtype=NP_FLOAT)


# The following code is from: <http://hetland.org/coding/python/levenshtein.py>
# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>
def levenshtein(a, b):
    """Calculate the Levenshtein distance between `a` and `b`.

    Args:
        a (str): Original word.
        b (str): Decoded word.

    Returns:
        float: Levenshtein distance.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
