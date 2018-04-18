"""Contains the TS model definition."""

import tensorflow as tf
from tensorflow import contrib as tfc

from params import FLAGS, TF_FLOAT
import utils
import s_input


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
        # stack = tf.nn.rnn_cell.MultiRNNCell([create_cell(num_hidden) for _ in range(num_layers)])
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
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
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
        # Not installed at the moment.
        # https://github.com/baidu-research/warp-ctc
        total_loss = tfc.wrapctc.wrap_ctc_loss(labels=labels,
                                               inputs=logits,
                                               sequence_length=seq_length)
    else:
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
        total_loss = tf.nn.ctc_loss(labels=labels,
                                    inputs=logits,
                                    sequence_length=seq_length,
                                    preprocess_collapse_repeated=False,
                                    ctc_merge_repeated=True,
                                    time_major=True)

    mean_loss = tf.reduce_mean(total_loss)
    tf.summary.scalar('mean_loss', mean_loss)

    return mean_loss


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
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.num_examples_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # Compute gradients.    Review which optimizer performs best?
    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
    # optimizer = s_utils.AdamOptimizerLogger(learning_rate=lr, beta1=FLAGS.adam_beta1,
    #                                         beta2=FLAGS.adam_beta2, epsilon=FLAGS.adam_epsilon)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    tf.summary.scalar('learning_rate', lr)

    return optimizer.minimize(_loss, global_step=global_step)


def decoding(logits, seq_len, labels, originals):
    """Decode a given inference (`logits`) and calculate the edit distance and the word error rate.

    Args:
        logits (tf.Tensor): Logits Tensor of shape [time (input), batch_size, num_classes].
        seq_len (tf.Tensor): Tensor containing the batches sequence lengths of shape [batch_size].
        labels (tf.SparseTensor): Integer SparseTensor containing the target.
            With dense shape [batch_size, time (target)].
        originals (tf.Tensor): String Tensor of shape [batch_size] with the original plaintext.

    Returns:
        tf.Tensor: Float Mean Edit Distance.
        tf.Tensor: Float Word Error Rate.
    """
    # Review label_len needed, instead of seq_len?

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

    # Edit distance and label error rate (LER).
    edit_distances = tf.edit_distance(decoded, labels)
    tf.summary.histogram('edit_distances', edit_distances)

    mean_edit_distance = tf.reduce_mean(edit_distances)
    tf.summary.scalar('mean_edit_distance', mean_edit_distance)

    # Translate decoded integer data back to character strings.
    dense = tf.sparse_tensor_to_dense(decoded)
    decoded_text_summary, decoded_texts = tf.py_func(utils.dense_to_text,
                                                     [dense, originals],
                                                     [tf.string, tf.string],
                                                     name='py_dense_to_text')

    tf.summary.text('decoded_text_summary', decoded_text_summary[:, : FLAGS.num_samples_to_report])

    # Word Error Rate (WER)
    wers, wer = tf.py_func(utils.wer_batch, [originals, decoded_texts], [TF_FLOAT, TF_FLOAT],
                           name='py_wer_batch')
    tf.summary.histogram('word_error_rates', wers)
    tf.summary.scalar('word_error_rate', wer)

    return mean_edit_distance, wer


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


def inputs():
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
    sequences, seq_length, labels, originals = s_input.inputs(FLAGS.batch_size)
    return sequences, seq_length, labels, originals


def _variable_on_cpu(name, shape, initializer):
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


def _variable_with_weight_decay(name, shape, stddev, weight_decay):
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
    var = _variable_on_cpu(name, shape, initializer=initializer)

    if weight_decay is not None:
        wd = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    return var
