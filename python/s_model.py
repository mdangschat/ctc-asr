"""Contains the TS model definition."""

import tensorflow as tf
import tensorflow.contrib as tfc

import s_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")

# Global constants describing the data set.
NUM_CLASSES = s_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_TRAIN = s_input.NUM_EXAMPLES_PER_EPOCH_TRAIN

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 0.33          # review Number of epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.75    # review Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01         # review Initial learning rate.


def inference(sample_batch, length_batch):
    """Build the TS model.
    # review Documentation

    Args:
        length_batch ():
        sample_batch ():

    Returns:
        tf.Tensor:
            Softmax layer pre activation function, i.e. layer(X*W + b)
    """
    num_hidden = 128
    print('inference:', sample_batch, length_batch)
    # LSTM cells
    with tf.variable_scope('lstm'):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=128, state_is_tuple=True)    # review: test this
        # cell = tfc.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
        num_layers = 1
        stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  # review
        # stack = tfc.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        # The second output is the last hidden state, it's not required anymore.
        cell_out, _ = tf.nn.dynamic_rnn(stack, sample_batch, sequence_length=length_batch, dtype=tf.float32)

        print('cell_out:', cell_out)
        cell_out = tf.reshape(cell_out, [-1, num_hidden])
        print('cell_out.reshape:', cell_out)

    # linear layer(XW + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('logits') as scope:
        weights = _variable_with_weight_decay('weights', [num_hidden, NUM_CLASSES], 0.04, 0.004)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(cell_out, weights), biases, name=scope.name)

        batch_size = tf.shape(sample_batch)[0]
        print('logits:', logits, ', batch_size:', batch_size)
        logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
        logits = tf.transpose(logits, [1, 0, 2])
        # _activation_summary(logits)

    return logits


def loss(logits, label_batch, length_batch, batch_size=FLAGS.batch_size):
    """L8ER Documentation

    Args:
        logits (tf.Tensor):
            3-D float Tensor. If time_major == False, this will be a Tensor shaped:
            [batch_size, max_time, num_classes]. If time_major == True (default), this will be a
            Tensor shaped: [max_time, batch_size, num_classes]. The logits.

        label_batch (tf.SparseTensor):
            An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the
            id for (batch b, time t). labels.values[i] must take on values in [0, num_labels).

        length_batch (tf.Tensor):
            1-D int32 vector, size [batch_size]. The sequence lengths.

        batch_size (int): TODO: FLAGS.batch_size could be wrong, since we allow smaller final batches.
            Batch size.

    Returns:
        A 1-D float Tensor, size [batch], containing the negative log probabilities.
    """
    print('shape1:', tf.shape(label_batch), label_batch)
    # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
    label_batch = tfc.layers.dense_to_sparse(label_batch)
    print('shape2:', tf.shape(label_batch), label_batch)

    # Reshape logits for CTC loss.
    logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])
    # Logits time major.
    logits = tf.transpose(logits, [1, 0, 2])

    print('ctc_loss:', label_batch, logits, length_batch)

    # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
    losses = tf.nn.ctc_loss(labels=label_batch, inputs=logits, sequence_length=length_batch)
    tf.summary.histogram('losses', losses)
    mean_loss = tf.reduce_mean(losses, name='mean_loss')
    tf.summary.scalar('mean_loss', mean_loss)
    return losses


def train(total_loss, global_step):
    """Train the TS model.
    L8ER documentation

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from the loss() function.
        global_step: Variable counting the number of training steps processed.

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
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer.minimize(total_loss, global_step=global_step)


def inputs_train():
    """Construct modified input for the TS training.
    review Documentation

    Returns:
        samples: Image 4D tf.Tensor of [batch_size, width, height, channels] size.
        label_batch: Labels 1D tf.Tensor of [batch_size] size.
    """
    sample_batch, label_batch, length_batch = s_input.inputs_train(FLAGS.batch_size)
    return sample_batch, label_batch, length_batch


def inputs():
    # L8ER: Write according to `inputs_train`.
    raise NotImplementedError


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
