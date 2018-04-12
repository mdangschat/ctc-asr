"""Utility and helper methods for TensorFlow speech learning."""

import numpy as np
from git import Repo
import tensorflow as tf

from s_labels import itoc


class AdamOptimizerLogger(tf.train.AdamOptimizer):
    # TODO: Document
    def _apply_dense(self, grad, var):
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta1_power, beta2_power = self._get_beta_accumulators()

        m_hat = m / (1. - beta1_power)
        v_hat = v / (1. - beta2_power)

        step = m_hat / (v_hat ** 0.5 + self._epsilon_t)

        # Use a histogram summary to monitor it during training.
        tf.summary.histogram('hist', step)

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
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
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
    repo = Repo('.', search_parent_directories=True)
    return repo.head.object.hexsha


def get_git_branch():
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
    # review Can be: tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell, tf.nn.rnn_cell.LSTMCell
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


def dense_to_text(decoded_batch, original_batch):
    # L8ER Documentation
    # L8ER Move somewhere else?
    decoded_result = ['"']
    original_result = ['"']

    for _decoded in decoded_batch:
        for i in _decoded:
            decoded_result.append(itoc(i))
        decoded_result.append('", "')

    for _original in original_batch:
        _original = str(_original, 'utf-8')
        for c in _original:
            original_result.append(c)
        original_result.append('", "')

    decoded_result = ''.join(decoded_result)[: -3]
    original_result = ''.join(original_result)[: -3]
    print('d: {}\no: {}'.format(decoded_result, original_result))

    decoded_result = np.array(decoded_result, dtype=np.object)
    original_result = np.array(original_result, dtype=np.object)
    return np.vstack([decoded_result, original_result])
