"""Utility and helper methods for TensorFlow speech learning."""

import numpy as np
from git import Repo
import tensorflow as tf

from s_labels import itoc
from s_params import FLAGS, NP_FLOAT


class AdamOptimizerLogger(tf.train.AdamOptimizer):
    """Modified `AdamOptimizer`_ that logs it's learning rate and step.

    .. _AdamOptimizer:
        https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    """

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta1_power, beta2_power = self._get_beta_accumulators()

        m_hat = m / (1. - beta1_power)
        v_hat = v / (1. - beta2_power)

        step = m_hat / (v_hat ** 0.5 + self._epsilon_t)

        # Use a histogram summary to monitor it during training.
        tf.summary.histogram('step', step)

        current_lr = self._lr_t * tf.sqrt(1. - beta2_power) / (1. - beta1_power)
        tf.summary.scalar('estimated_lr', current_lr)

        return super(AdamOptimizerLogger, self)._apply_dense(grad, var)


def attention(inputs, attention_size, time_major=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw)
                containing the forward and the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    output = inputs * tf.expand_dims(alphas, -1)

    return output, alphas


def get_git_revision_hash():
    """Return the git revision id/hash.

    Returns:
        str: Git revision hash.
    """
    repo = Repo('.', search_parent_directories=True)
    return repo.head.object.hexsha


def get_git_branch():
    """Return the active git branches name.

    Returns:
        str: Git branch.
    """
    repo = Repo('.', search_parent_directories=True)
    return repo.active_branch.name


def create_cell(num_units, keep_prob=1.0):
    """Create a RNN cell with added dropout wrapper.

    Args:
        num_units (int): Number of units within the RNN cell.
        keep_prob (float): Probability [0, 1] to keep an output. It it's constant 1
            no outputs will be dropped.

    Returns:
        tf.nn.rnn_cell.LSTMCell: RNN cell with dropout wrapper.
    """
    # Can be: tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell, tf.nn.rnn_cell.LSTMCell
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def create_bidirectional_cells(num_units, _num_layers, keep_prob=1.0):
    """Create two lists of forward and backward cells that can be used to build
    a BDLSTM stack.

    Args:
        num_units (int): Number of units within the RNN cell.
        _num_layers (int): Amount of cells to create for each list.
        keep_prob (float): Probability [0, 1] to keep an output. It it's constant 1
            no outputs will be dropped.

    Returns:
        [tf.nn.rnn_cell.LSTMCell]: List of forward cells.
        [tf.nn.rnn_cell.LSTMCell]: List of backward cells.
    """
    _fw_cells = [create_cell(num_units, keep_prob=keep_prob) for _ in range(_num_layers)]
    _bw_cells = [create_cell(num_units, keep_prob=keep_prob) for _ in range(_num_layers)]
    return _fw_cells, _bw_cells


def dense_to_text(decoded, originals):
    """Convert a dense, integer encoded `tf.Tensor` into a readable string.

        Args:
            decoded (tf.Tensor):
                The decoded integer Tensor path.
            originals (tf.Tensor):
                String tensor, containing the original input string for comparision.

        Returns:
            tf.Tensor:
                2D string Tensor with layout:
                    [[decoded_string_0, original_string_0], ...
                     [decoded_string_N, original_string_N]]
            tf.Tensor:
                1D string Tensor containing only the decoded text outputs.
                    [decoded_string_0, ..., decoded_string_N]
        """
    decoded_strings = []
    original_strings = []

    for d in decoded:
        decoded_strings.append(''.join([itoc(i) for i in d]))

    for o in originals:
        original_strings.append(''.join([c for c in o.decode('utf-8')]))

    # Print a maximum of `FLAGS.num_samples_to_report` to STDOUT.   # TODO deactivated for eval rewr
    # print('d: {}\no: {}'.format(decoded_strings[: FLAGS.num_samples_to_report],
    #                             original_strings[: FLAGS.num_samples_to_report]))

    decoded_strings = np.array(decoded_strings, dtype=np.object)
    original_strings = np.array(original_strings, dtype=np.object)
    return np.vstack([decoded_strings, original_strings]), np.array(decoded_strings)


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
