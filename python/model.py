"""Contains the TS model definition."""

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

import warpctc_tensorflow as warpctc

from python.params import FLAGS, TF_FLOAT
import python.utils as utils
import python.s_input as s_input


def inference(sequences, seq_length):
    """Build the speech model.

    Args:
        sequences (tf.Tensor): 3D float Tensor with input sequences. [batch_size, time, NUM_INPUTS]
        seq_length (tf.Tensor): 1D int Tensor with sequence length. [batch_size]

    Returns:
        tf.Tensor:
            Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
    """
    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=TF_FLOAT)
    regularizer = tfc.layers.l2_regularizer(0.0046875)

    # Dense1
    with tf.variable_scope('dense1'):
        dense1 = tf.layers.dense(sequences, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer)
        dense1 = tf.minimum(dense1, FLAGS.relu_cutoff)

    # Dense2
    with tf.variable_scope('dense2'):
        dense2 = tf.layers.dense(dense1, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense2 = tf.minimum(dense2, FLAGS.relu_cutoff)

    # Dense3
    with tf.variable_scope('dense3'):
        dense3 = tf.layers.dense(dense2, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense3 = tf.minimum(dense3, FLAGS.relu_cutoff)

    # BDLSTM cell stack.
    with tf.variable_scope('bdlstm'):
        # Create a stack of RNN cells.
        fw_cells, bw_cells = utils.create_bidirectional_cells(FLAGS.num_units_lstm,
                                                              FLAGS.num_layers_lstm,
                                                              keep_prob=1.0)

        # `output` = [batch_size, time, num_hidden*2]
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
        bdlstm, _, _ = tfc.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells,
                                                               inputs=dense3,
                                                               dtype=TF_FLOAT,
                                                               sequence_length=seq_length,
                                                               parallel_iterations=64,  # review
                                                               time_major=False)

    # Dense4
    with tf.variable_scope('dense4'):
        dense4 = tf.layers.dense(bdlstm, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense4 = tf.minimum(dense4, FLAGS.relu_cutoff)

    # Logits: layer(XW + b),
    # We don't apply softmax here because most TensorFlow loss functions perform
    # a softmax activation as needed, and therefore don't expect activated logits.
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(dense4, FLAGS.num_classes, kernel_initializer=initializer)
        logits = tfc.rnn.transpose_batch_time(logits)

    # `logits` = [time, batch_size, NUM_CLASSES]
    return logits


def loss(logits, labels, seq_length):
    """Calculate the networks CTC loss.

    Args:
        logits (tf.Tensor):
            3D float Tensor. If time_major == False, this will be a Tensor shaped:
            [batch_size, max_time, num_classes]. If time_major == True (default), this will be a
            Tensor shaped: [max_time, batch_size, num_classes]. The logits.

        labels (tf.SparseTensor):
            An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the
            id for (batch b, time t). labels.values[i] must take on values in [0, num_labels).

        seq_length (tf.Tensor):
            1D int32 vector, size [batch_size]. The sequence lengths.

    Returns:
        tf.Tensor:
            1D float Tensor with size [1], containing the mean loss.
    """
    if FLAGS.use_warp_ctc:
        # https://github.com/baidu-research/warp-ctc
        flat_labels = tf.sparse_tensor_to_dense(labels)
        flat_labels = tf.reshape(flat_labels, [-1])
        flat_labels = tf.Print(flat_labels, [flat_labels], message='flat_labels ')
        seq_length = tf.Print(seq_length, [seq_length], message='seq_length ')

        # TODO: label_lengths is a placeholder!
        total_loss = warpctc.ctc(activations=logits,
                                 flat_labels=flat_labels,
                                 label_lengths=seq_length,
                                 input_lengths=seq_length)

    else:
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
        total_loss = tf.nn.ctc_loss(labels=labels,
                                    inputs=logits,
                                    sequence_length=seq_length,
                                    preprocess_collapse_repeated=False,
                                    ctc_merge_repeated=True,
                                    time_major=True)

    # Return average CTC loss.
    return tf.reduce_mean(total_loss)


def decode(logits, seq_len, originals=None):
    """Decode a given inference (`logits`) and convert it to plaintext.

    Args:
        logits (tf.Tensor):
            Logits Tensor of shape [time (input), batch_size, num_classes].
        seq_len (tf.Tensor):
            Tensor containing the batches sequence lengths of shape [batch_size].
        originals (tf.Tensor): Optional, default `None`.
            String Tensor of shape [batch_size] with the original plaintext.

    Returns:
        tf.Tensor: Decoded integer labels.
        tf.Tensor: Decoded plaintext's.
        tf.Tensor: Decoded plaintext's and original texts for comparision in `tf.summary.text`.
    """
    # tf.nn.ctc_beam_search_decoder provides more accurate results, but is slower.
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
    # decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_len)
    decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                               sequence_length=seq_len,
                                               beam_width=FLAGS.beam_width,
                                               top_paths=1,
                                               merge_repeated=False)

    # ctc_greedy_decoder returns a list with one SparseTensor as only element, if `top_paths=1`.
    decoded = tf.cast(decoded[0], tf.int32)

    dense = tf.sparse_tensor_to_dense(decoded)

    originals = originals if originals is not None else np.array([], dtype=np.int32)

    # Translate decoded integer data back to character strings.
    plaintext, plaintext_summary = tf.py_func(utils.dense_to_text, [dense, originals],
                                              [tf.string, tf.string], name='py_dense_to_text')

    return decoded, plaintext, plaintext_summary


def decoded_error_rates(labels, originals, decoded, decoded_texts):
    """Calculate edit distance and word error rate.

    Args:
        labels (tf.SparseTensor):
            Integer SparseTensor containing the target.
            With dense shape [batch_size, time (target)].
        originals (tf.Tensor):
            String Tensor of shape [batch_size] with the original plaintext.
        decoded (tf.Tensor):
            Integer tensor of the decoded output labels.
        decoded_texts (tf.Tensor)
            String tensor with the decoded output labels converted to normal text.

    Returns:
        tf.Tensor: Edit distances for the batch.
        tf.Tensor: Mean edit distance.
        tf.Tensor: Word error rates for the batch.
        tf.Tensor: Word error rate.
    """
    # Edit distances and average edit distance.
    edit_distances = tf.edit_distance(decoded, labels)
    mean_edit_distance = tf.reduce_mean(edit_distances)

    # Word error rates for the batch and average word error rate (WER).
    wers, wer = tf.py_func(utils.wer_batch, [originals, decoded_texts], [TF_FLOAT, TF_FLOAT],
                           name='py_wer_batch')

    return edit_distances, mean_edit_distance, wers, wer


def train(_loss, global_step):
    """Train op for the speech model.

    Create an optimizer and apply to all trainable variables.

    Args:
        _loss (tf.Tensor):
            Scalar Tensor of type float containing total loss from the loss() function.
        global_step (tf.Tensor):
            Scalar Tensor of type int32 counting the number of training steps processed.

    Returns:
        tf.Tensor:
            Optimizer operator for training.
    """
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    FLAGS.steps_per_decay,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Select a gradient optimizer.
    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
    # optimizer = s_utils.AdamOptimizerLogger(learning_rate=lr, beta1=FLAGS.adam_beta1,
    #                                         beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    tf.summary.scalar('learning_rate', lr)

    return optimizer.minimize(_loss, global_step=global_step)


def inputs_train():
    """Construct input for the speech training.

    Returns:
        tf.Tensor:
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor:
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor:
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor:
            2D Tensor with the original strings.
    """
    sequences, seq_length, labels, originals = s_input.inputs_train(FLAGS.batch_size)
    return sequences, seq_length, labels, originals


def inputs(target='test'):
    """Construct input for the speech training.

    Args:
        target (str): 'train' or 'validate'.

    Returns:
        tf.Tensor:
            3D Tensor with sequence batch of shape [batch_size, time, data].
            Where time is equal to max(seq_len) for the bucket batch.
        tf.Tensor:
            1D Tensor with sequence lengths for each sequence within the batch.
            With shape [batch_size], and type tf.int32.
        tf.Tensor:
            2D Tensor with labels batch of shape [batch_size, max_label_len],
            with max_label_len equal to max(len(label)) for the bucket batch.
            Type is tf.int32.
        tf.Tensor:
            2D Tensor with the original strings.
    """
    if target != 'test' and target != 'validate':
        raise ValueError('"{}" is not a valid target.'.format(target))

    sequences, seq_length, labels, originals = s_input.inputs(FLAGS.batch_size)
    return sequences, seq_length, labels, originals
