"""Contains the TS model definition."""

import numpy as np
import tensorflow as tf

import s_input
import s_labels


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """(Maximum) Number of samples within a batch.""")

# Global constants describing the data set.
NUM_CLASSES = s_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_TRAIN = s_input.NUM_EXAMPLES_PER_EPOCH_TRAIN

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 0.4          # Number of epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.50   # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.


def inference(sequences, seq_length):
    """Build the speech model.

    Args:
        sequences (tf.Tensor): 3D Tensor with input sequences.
        seq_length (tf.Tensor): 2D Tensor with sequence length.

    Returns:
        tf.Tensor:
            Softmax layer (logits) pre activation function, i.e. layer(X*W + b)
    """
    # LSTM cells
    num_hidden = 128
    num_layers = 2

    def create_cell(num_units, keep_prob=1.0):
        """Create a RNN cell with added dropout wrapper.

        Args:
            num_units (int): Number of units within the RNN cell.
            keep_prob (float): Probability [0, 1] to keep an output. It it's constant 1
                no outputs will be dropped.

        Returns:
            tf.LayerRNNCell:
                RNN cell with dropout wrapper.
        """
        # review Can be: tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell, tf.nn.rnn_cell.LSTMCell
        cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, use_peepholes=True)
        drop = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return drop

    with tf.variable_scope('rnn'):
        # Create a stack of RNN cells.
        # stack = tf.nn.rnn_cell.MultiRNNCell([create_cell(num_hidden) for _ in range(num_layers)])
        fw1, bw1 = create_cell(num_hidden), create_cell(num_hidden)

        # batch_size = tf.shape(seq_length)[0]
        # sequences = tf.Print(sequences, [tf.shape(sequences)], message='sequences: ')
        # initial_state = stack.zero_state(batch_size, dtype=tf.float32)
        # `sequences` [batch_size, time, data]
        # The second output is the final hidden state, it's not required anymore.
        # cell_out, _ = tf.nn.dynamic_rnn(cell=stack,
        #                                 inputs=sequences,
        #                                 sequence_length=seq_length,
        #                                 initial_state=initial_state,
        #                                 dtype=tf.float32)

        # `cell_out` [batch_size, time, num_hidden]
        cell_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw1, cell_bw=bw1,
                                                      inputs=sequences,
                                                      sequence_length=seq_length,
                                                      dtype=tf.float32)
        cell_out = tf.concat([cell_out[0], cell_out[1]], 2)

        # cell_out = tf.Print(cell_out, [tf.shape(cell_out), tf.shape(_)], message='cell_out: ')

        # Reshape for dense layer.
        # cell_out = tf.reshape(cell_out, [-1, num_hidden * 2])

    # Logits: layer(XW + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('logits') as scope:
        # weights = _variable_with_weight_decay('weights', [num_hidden * 2, NUM_CLASSES], 0.04, 0.004)
        # biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        # logits = tf.add(tf.matmul(cell_out, weights), biases, name=scope.name)
        #
        # batch_size = tf.shape(sequences)[0]
        # logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
        # logits = tf.transpose(logits, [1, 0, 2])
        # `logits` [time, batch_size, NUM_CLASSES]
        # _activation_summary(logits)
        logits = tf.layers.dense(cell_out, NUM_CLASSES,
                                 kernel_initializer=tf.glorot_normal_initializer())
        logits = tf.transpose(logits, [1, 0, 2])

    # logits = tf.Print(logits, [tf.shape(logits)], message='logits: ')
    return logits


def loss(logits, labels, seq_length):
    """Calculate the networks loss.

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
    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    losses = tf.nn.ctc_loss(labels=labels,
                            inputs=logits,
                            sequence_length=seq_length,
                            preprocess_collapse_repeated=False,
                            ctc_merge_repeated=True,
                            time_major=True)

    mean_loss = tf.reduce_mean(losses)
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
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Compute gradients. review Optimizers
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
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
    sample_batch, label_batch, length_batch, originals_batch = s_input.inputs_train(
        FLAGS.batch_size)
    return sample_batch, label_batch, length_batch, originals_batch


def inputs():
    """Construct input for the speech evaluation.

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
    """
    sample_batch, label_batch, length_batch = s_input.inputs(FLAGS.batch_size)
    return sample_batch, label_batch, length_batch


def decoding(logits, seq_len, labels, originals):
    # TODO: Implement & Document
    # Review label_len needed, instead of seq_len?

    def dense_to_text(decoded_batch, original_batch):
        # L8ER Documentation
        # L8ER Move somewhere else?
        decoded_result = ['"']
        original_result = ['"']

        for _decoded in decoded_batch:
            for i in _decoded:
                decoded_result.append(s_labels.itoc(i))
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

    print('decoding:', logits, ', ', seq_len, ', ', labels)
    # Review: tf.nn.ctc_beam_search_decoder provides more accurate results, but is slower.
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len)
    decoded = decoded[0]    # ctc_greedy_decoder returns a list with 1 SparseTensor as only element.
    print('ctc_greedy_decoder:', decoded, log_prob)
    # seq_len = tf.Print(seq_len, [seq_len, decoded.dense_shape, log_prob], message='ctc_greedy_decoder: ')
    tf.summary.histogram('delete_me', seq_len)

    # Edit distance and label error rate (LER).
    edit_distance = tf.edit_distance(tf.cast(decoded, tf.int32), labels)
    tf.summary.histogram('edit_distance', edit_distance)
    label_error_rate = tf.reduce_mean(edit_distance)
    # label_error_rate = tf.Print(label_error_rate, [label_error_rate, edit_distance], message='ler & ed: ')
    tf.summary.scalar('label_error_rate', label_error_rate)

    # review: Experimental decoding
    dense = tf.sparse_tensor_to_dense(decoded)
    # dense = tf.Print(dense, [dense, tf.shape(dense)], message='dense: ', summarize=100)
    text = tf.py_func(dense_to_text, [dense, originals], tf.string)
    text = tf.cast(text, dtype=tf.string)
    # dense = tf.Print(dense, [text, tf.shape(text)], message='text: ', summarize=100)

    tf.summary.text('decoded_text', text)

    return label_error_rate


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor

    Returns:
        nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a variable stored on CPU memory.

    Args:
        name (str): Name of the variable.
        shape (list of int): List of ints, e.g. a numpy shape.
        initializer: Initializer for the variable.

    Returns:
        Variable tensor.
    """
    with tf.device('/cpu:0'):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


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
        Variable tensor.
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if weight_decay is not None:
        wd = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)
    return var
