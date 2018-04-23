"""Utility and helper methods for TensorFlow speech learning."""

import time
from datetime import datetime
import numpy as np
from git import Repo
import tensorflow as tf

from params import FLAGS, NP_FLOAT, TF_FLOAT
from s_labels import itoc


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
    return tf.nn.rnn_cell.DropoutWrapper(cell,
                                         input_keep_prob=keep_prob,
                                         output_keep_prob=keep_prob,
                                         seed=FLAGS.random_seed)


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


def variable_on_cpu(name, shape, initializer):
    """Helper to create a variable stored on CPU memory.

    Args:
        name (str): Name of the variable.
        shape (list of int): List of ints, e.g. a numpy shape.
        initializer: Initializer for the variable.

    Returns:
        tf.Tensor: Variable tensor.
    """
    with tf.device('/cpu:0'):
        return tf.get_variable(name, shape, initializer=initializer, dtype=TF_FLOAT)


def variable_with_weight_decay(name, shape, stddev, weight_decay):
    """Helper to create an initialized variable with weight decay.

    Note that the variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name (str): Name of the variable.
        shape (list of int): List of ints, e.g. a numpy shape.
        stddev (float): Standard deviation of the Gaussian.
        weight_decay: Add L2Loss weight decay multiplied by this float.
            If None, weight decay is not added for this variable.

    Returns:
        tf.Tensor: Variable tensor.
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=TF_FLOAT)
    var = variable_on_cpu(name, shape, initializer=initializer)

    if weight_decay is not None:
        wd = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    return var


# The following code has been inspired by <https://stackoverflow.com/a/45681782>.
class TraceHook(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(self, file_writer, log_frequency, trace_level=tf.RunOptions.FULL_TRACE):
        self._trace = log_frequency == 1
        self.writer = file_writer
        self.trace_level = trace_level
        self.log_frequency = log_frequency
        self._global_step_tensor = None

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use TraceHook.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None

        return tf.train.SessionRunArgs(fetches=self._global_step_tensor, options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results - 1
        if self._trace:
            self._trace = False
            self.writer.add_run_metadata(run_values.run_metadata, '{}'.format(global_step))
        if not (global_step + 1) % self.log_frequency:
            self._trace = True


class LoggerHook(tf.train.SessionRunHook):
    """Log loss and runtime."""

    def __init__(self, loss_op):
        self.loss_op = loss_op
        self._global_step_tensor = None
        self._start_time = 0

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        self._start_time = time.time()

    def before_run(self, run_context):
        # Asks for loss value and global step.
        return tf.train.SessionRunArgs(fetches=[self.loss_op, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        loss_value, global_step = run_values.results

        if global_step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = duration / float(FLAGS.log_frequency)
            batch_per_sec = float(FLAGS.log_frequency) / duration

            print('{}: step {:,d}, loss={:.4f}, {:.1f} examples/sec ({:.3f} sec/batch) '
                  '({:.1f} batch/sec)'.format(datetime.now(), global_step, loss_value,
                                              examples_per_sec, sec_per_batch, batch_per_sec))
