"""Contains the TS model definition."""

import tensorflow as tf
import tensorflow.contrib as tfc

import s_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")

# Global constants describing the data set.
NUM_CLASSES = s_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_TRAIN = s_input.NUM_EXAMPLES_PER_EPOCH_TRAIN

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 0.50          # Number of epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.75    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001        # Initial learning rate.


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
    num_layers = 1

    with tf.variable_scope('lstm'):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
        stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        # The second output is the last hidden state, it's not required anymore.
        cell_out, _ = tf.nn.dynamic_rnn(stack, sequences, sequence_length=seq_length,
                                        dtype=tf.float32)
        # Reshape for dense layer.
        cell_out = tf.reshape(cell_out, [-1, num_hidden])

    # Logits: layer(XW + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('logits') as scope:
        weights = _variable_with_weight_decay('weights', [num_hidden, NUM_CLASSES], 0.04, 0.004)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(cell_out, weights), biases, name=scope.name)

        batch_size = tf.shape(sequences)[0]
        logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
        logits = tf.transpose(logits, [1, 0, 2])
        # _activation_summary(logits)

    return logits


def loss(logits, labels, seq_length, batch_size=FLAGS.batch_size):
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

        batch_size (int): Batch size.
            Note that the default `FLAGS.batch_size` could be wrong,
            if we allow for smaller final batches.

    Returns:
        tf.Tensor:
            1D float Tensor with size [1], containing the mean loss.
    """
    # Reshape labels for CTC loss.
    # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
    label_batch_s = tfc.layers.dense_to_sparse(labels)

    # Reshape logits for CTC loss.
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
    # Logits time major.
    logits = tf.transpose(logits, [1, 0, 2])

    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    losses = tf.nn.ctc_loss(labels=label_batch_s,
                            inputs=logits,
                            sequence_length=seq_length,
                            preprocess_collapse_repeated=False,
                            ctc_merge_repeated=True,
                            time_major=True)

    tf.summary.histogram('losses', losses)
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

    # Compute gradients.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
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
    """
    sample_batch, label_batch, length_batch = s_input.inputs_train(FLAGS.batch_size)
    return sample_batch, label_batch, length_batch


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
    sample_batch, label_batch, length_batch = s_input.inputs_train(FLAGS.batch_size)
    return sample_batch, label_batch, length_batch


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
