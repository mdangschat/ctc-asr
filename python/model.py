"""Contains the TS model definition."""

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from python.params import FLAGS, TF_FLOAT
from python.util import tf_contrib, cost_metrics
import python.s_input as s_input

if FLAGS.use_warp_ctc:
    # noinspection PyUnresolvedReferences
    import warpctc_tensorflow as warp_ctc


def inference(sequences, seq_length, training=True):
    """Build a TensorFlow inference graph according to the selected model in `FLAGS.used_model`.
    Supports the default [Deep Speech 1] model ('ds1') and an [Deep Speech 2] inspired
    implementation ('ds2').

    Args:
        sequences (tf.Tensor):
            3D float Tensor with input sequences. [batch_size, time, NUM_INPUTS]
        seq_length (tf.Tensor):
            1D int Tensor with sequence length. [batch_size]
        training (bool):
            If `True` apply dropout else if `False` the data is passed through unaltered.

    Returns:
        tf.Tensor: `logits`
            Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
        tf.Tensor: `seq_length`
            1D Tensor containing approximated sequence lengths.
    """
    if FLAGS.used_model == 'ds1':
        return inference_ds1(sequences, seq_length, training=training)
    elif FLAGS.used_model == 'ds2':
        # DS2 convolutional layers don't need the `seq_length` parameter.
        return inference_ds2(sequences, training=training)
    else:
        raise ValueError('Unsupported model: {}'.format(FLAGS.used_model))


def inference_ds1(sequences, seq_length, training=True):
    """Build the asr model, mostly based on the original DS1 paper.

    Args:
        sequences (tf.Tensor):
            3D float Tensor with input sequences. [batch_size, time, NUM_INPUTS]
        seq_length (tf.Tensor):
            1D int Tensor with sequence length. [batch_size]
        training (bool):
            If `True` apply dropout else if `False` the data is passed through unaltered.

    Returns:
        tf.Tensor: `logits`
            Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
        tf.Tensor: `seq_length`
            1D Tensor containing approximated sequence lengths.
    """
    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=TF_FLOAT)
    regularizer = tfc.layers.l2_regularizer(0.0046875)

    # Dense1
    with tf.variable_scope('dense1'):
        # sequences = [batch_size, time, NUM_FEATURES]
        dense1 = tf.layers.dense(sequences, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 kernel_regularizer=regularizer)
        dense1 = tf.minimum(dense1, FLAGS.relu_cutoff)
        dense1 = tf.layers.dropout(dense1, rate=FLAGS.dense_dropout_rate, training=training)
        # dense1 = [batch_size, time, num_units_dense]

    # Dense2
    with tf.variable_scope('dense2'):
        dense2 = tf.layers.dense(dense1, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense2 = tf.minimum(dense2, FLAGS.relu_cutoff)
        dense2 = tf.layers.dropout(dense2, rate=FLAGS.dense_dropout_rate, training=training)
        # dense2 = [batch_size, time, num_units_dense]

    # Dense3
    with tf.variable_scope('dense3'):
        dense3 = tf.layers.dense(dense2, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense3 = tf.minimum(dense3, FLAGS.relu_cutoff)
        dense3 = tf.layers.dropout(dense3, rate=FLAGS.dense_dropout_rate, training=training)
        # dense3 = [batch_size, time, num_units_dense]

    # RNN layers.
    with tf.variable_scope('rnn'):
        dropout_rate = FLAGS.rnn_dropout_rate if training else 0.0

        if not FLAGS.use_cudnn:
            # Create a stack of RNN cells.
            fw_cells, bw_cells = tf_contrib.bidirectional_cells(FLAGS.num_units_rnn,
                                                                FLAGS.num_layers_rnn,
                                                                dropout=dropout_rate)

            # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
            rnn_output, _, _ = tfc.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells,
                                                                       inputs=dense3,
                                                                       dtype=TF_FLOAT,
                                                                       sequence_length=seq_length,
                                                                       parallel_iterations=64,
                                                                       time_major=False)
            # rnn_output = [batch_size, time, num_units_rnn * 2]

        else:   # FLAGS.use_cudnn
            # cuDNN RNNs only support time major inputs.
            conv_output = tfc.rnn.transpose_batch_time(dense3)

            # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnRNNTanh
            rnn = tfc.cudnn_rnn.CudnnRNNRelu(num_layers=FLAGS.num_layers_rnn,
                                             num_units=FLAGS.num_units_rnn,
                                             input_mode='linear_input',
                                             direction='bidirectional',
                                             dropout=dropout_rate,
                                             seed=FLAGS.random_seed,
                                             dtype=TF_FLOAT,
                                             kernel_initializer=None,   # Glorot Uniform Initializer
                                             bias_initializer=None)     # Constant 0.0 Initializer

            rnn_output, _ = rnn(conv_output)
            rnn_output = tfc.rnn.transpose_batch_time(rnn_output)
            # rnn_output = [batch_size, time, num_units_rnn * 2]

    # Dense4
    with tf.variable_scope('dense4'):
        dense4 = tf.layers.dense(rnn_output, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense4 = tf.minimum(dense4, FLAGS.relu_cutoff)
        dense4 = tf.layers.dropout(dense4, rate=FLAGS.dense_dropout_rate, training=training)
        # dense4 = [batch_size, conv_time, num_units_dense]

    # Logits: layer(XW + b),
    # We don't apply softmax here because most TensorFlow loss functions perform
    # a softmax activation as needed, and therefore don't expect activated logits.
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(dense4, FLAGS.num_classes, kernel_initializer=initializer)
        logits = tfc.rnn.transpose_batch_time(logits)

    # logits = [time, batch_size, NUM_CLASSES]
    return logits, seq_length


def inference_ds2(sequences, training=True):
    """Build the asr model.

    Args:
        sequences (tf.Tensor):
            3D float Tensor with input sequences. [batch_size, time, NUM_INPUTS]
        training (bool):
            If `True` apply dropout else if `False` the data is passed through unaltered.

    Returns:
        tf.Tensor: `logits`
            Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
        tf.Tensor: `seq_length`
            1D Tensor containing approximated sequence lengths.
    """
    initializer = tf.truncated_normal_initializer(stddev=0.046875, dtype=TF_FLOAT)
    regularizer = tfc.layers.l2_regularizer(0.0046875)

    # Convolutional layers.
    with tf.variable_scope('conv'):
        # sequences = [batch_size, time, NUM_INPUTS] => [batch_size, time, NUM_INPUTS, 1]
        sequences = tf.reshape(sequences, [sequences.shape[0], -1, sequences.shape[2], 1])

        # Apply convolutions.
        conv_output, seq_length = tf_contrib.conv_layers(sequences)

    # RNN layers.
    with tf.variable_scope('rnn'):
        dropout_rate = FLAGS.rnn_dropout_rate if training else 0.0

        if not FLAGS.use_cudnn:
            # Create a stack of RNN cells.
            fw_cells, bw_cells = tf_contrib.bidirectional_cells(FLAGS.num_units_rnn,
                                                                FLAGS.num_layers_rnn,
                                                                dropout=dropout_rate)

            # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
            rnn_output, _, _ = tfc.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells,
                                                                       inputs=conv_output,
                                                                       dtype=TF_FLOAT,
                                                                       sequence_length=seq_length,
                                                                       parallel_iterations=64,
                                                                       time_major=False)
            # rnn_output = [batch_size, time, num_units_rnn * 2]

        else:   # FLAGS.use_cudnn
            # cuDNN RNNs only support time major inputs.
            conv_output = tfc.rnn.transpose_batch_time(conv_output)

            # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnRNNRelu
            # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM
            # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnGRU
            rnn = tfc.cudnn_rnn.CudnnRNNRelu(num_layers=FLAGS.num_layers_rnn,
                                             num_units=FLAGS.num_units_rnn,
                                             input_mode='linear_input',
                                             direction='bidirectional',
                                             dropout=dropout_rate,
                                             seed=FLAGS.random_seed,
                                             dtype=TF_FLOAT,
                                             kernel_initializer=None,   # Glorot Uniform Initializer
                                             bias_initializer=None)     # Constant 0.0 Initializer

            rnn_output, _ = rnn(conv_output)
            rnn_output = tfc.rnn.transpose_batch_time(rnn_output)
            # rnn_output = [batch_size, time, num_units_rnn * 2]

    # Dense4
    with tf.variable_scope('dense4'):
        dense4 = tf.layers.dense(rnn_output, FLAGS.num_units_dense,
                                 activation=tf.nn.relu,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizer)
        dense4 = tf.minimum(dense4, FLAGS.relu_cutoff)
        dense4 = tf.layers.dropout(dense4, rate=FLAGS.dense_dropout_rate, training=training)
        # dense4 = [batch_size, conv_time, num_units_dense]

    # Logits: layer(XW + b),
    # We don't apply softmax here because most TensorFlow loss functions perform
    # a softmax activation as needed, and therefore don't expect activated logits.
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(dense4, FLAGS.num_classes, kernel_initializer=initializer)
        logits = tfc.rnn.transpose_batch_time(logits)

    # logits = [time, batch_size, NUM_CLASSES]
    return logits, seq_length


def loss(logits, seq_length, labels, label_length):
    """Calculate the networks CTC loss.

    Args:
        logits (tf.Tensor):
            3D float Tensor. If time_major == False, this will be a Tensor shaped:
            [batch_size, max_time, num_classes]. If time_major == True (default), this will be a
            Tensor shaped: [max_time, batch_size, num_classes]. The logits.

        labels (tf.SparseTensor or tf.Tensor):
            An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the
            id for (batch b, time t). labels.values[i] must take on values in [0, num_labels), if
            `FLAGS.use_warp_ctc` is false.
            Else, an int32 dense Tensor version of the above sparse version.

        seq_length (tf.Tensor):
            1D int32 vector, size [batch_size]. The sequence lengths.

        label_length (tf.Tensor):
            1D Tensor with the length of each label within the batch. Shape [batch_size].

    Returns:
        tf.Tensor:
            1D float Tensor with size [1], containing the mean loss.
    """
    if FLAGS.use_warp_ctc:
        # Labels need to be a 1D vector, with every label concatenated.
        flat_labels = tf.reshape(labels, [-1])

        # Remove padding from labels.
        partitions = tf.cast(tf.equal(flat_labels, 0), tf.int32)
        flat_labels, _ = tf.dynamic_partition(flat_labels, partitions, 2)

        # `label_length` needs to be a 1D vector.
        flat_label_length = tf.reshape(label_length, [-1])

        # https://github.com/baidu-research/warp-ctc
        total_loss = warp_ctc.ctc(activations=logits,
                                  flat_labels=flat_labels,
                                  label_lengths=flat_label_length,
                                  input_lengths=seq_length,
                                  blank_label=28)

        # total_loss = tf.Print(total_loss, [total_loss], message='total_loss ')

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

        originals (tf.Tensor or None): Optional, default `None`.
            String Tensor of shape [batch_size] with the original plaintext.

    Returns:
        tf.Tensor: Decoded integer labels.
        tf.Tensor: Decoded plaintext's.
        tf.Tensor: Decoded plaintext's and original texts for comparision in `tf.summary.text`.
    """
    # tf.nn.ctc_beam_search_decoder provides more accurate results, but is slower.
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
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
    plaintext, plaintext_summary = tf.py_func(cost_metrics.dense_to_text, [dense, originals],
                                              [tf.string, tf.string], name='py_dense_to_text')

    return decoded, plaintext, plaintext_summary


def decoded_error_rates(labels, originals, decoded, decoded_texts):
    """Calculate edit distance and word error rate.

    Args:
        labels (tf.SparseTensor or tf.Tensor):
            Integer SparseTensor containing the target.
            With dense shape [batch_size, time (target)].
            Dense Tensors are converted into SparseTensors if `FLAGS.use_warp_ctc == True`.

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
    if FLAGS.use_warp_ctc:
        labels = tfc.layers.dense_to_sparse(labels)

    # Edit distances and average edit distance.
    edit_distances = tf.edit_distance(decoded, labels)
    mean_edit_distance = tf.reduce_mean(edit_distances)

    # Word error rates for the batch and average word error rate (WER).
    wers, wer = tf.py_func(cost_metrics.wer_batch, [originals, decoded_texts],
                           [TF_FLOAT, TF_FLOAT], name='py_wer_batch')

    return edit_distances, mean_edit_distance, wers, wer


def train(_loss, global_step):
    """Train operator for the asr model.

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

    # Set the minimum learning rate.
    lr = tf.maximum(lr, FLAGS.minimum_lr)

    # Select a gradient optimizer.
    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    tf.summary.scalar('learning_rate', lr)

    return optimizer.minimize(_loss, global_step=global_step)


def inputs_train(shuffle):
    """Construct input for the asr training.

    Args:
        shuffle (bool): Shuffle data or not. See `s_input.inputs_train()`.

    Returns:
        See `s_input.inputs_train()`.
    """
    return s_input.inputs_train(FLAGS.batch_size, shuffle=shuffle)


def inputs(target):
    """Construct input for the asr evaluation.

    Args:
        target (str): 'train' or 'dev'.

    Returns:
        See `s_input.inputs()`.
    """
    if target != 'test' and target != 'dev':
        raise ValueError('"{}" is not a valid target.'.format(target))

    return s_input.inputs(FLAGS.batch_size, target)
