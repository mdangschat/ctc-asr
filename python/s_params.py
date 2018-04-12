"""Collection of hyper parameters."""
# TODO: Convert to FLAGs where applicable.


import tensorflow as tf
import numpy as np

from s_labels import num_classes


# Constants describing the training process.
tf.flags.DEFINE_integer('batch_size', 2,
                        """(Maximum) Number of samples within a batch.""")

tf.flags.DEFINE_float('learning_rate', 1e-3,
                      """Initial learning rate.""")
tf.flags.DEFINE_float('learning_rate_decay_factor', 0.75,
                      """Learning rate decay factor.""")
tf.flags.DEFINE_float('num_epochs_per_decay', 10.0,
                      """Number of epochs after which learning rate decays.""")

tf.flags.DEFINE_float('adam_beta1', 0.9,
                      """Adam optimizer beta_1 power.""")
tf.flags.DEFINE_float('adam_beta2', 0.999,
                      """Adam optimizer beta_2 power.""")
tf.flags.DEFINE_float('adam_epsilon', 1e-8,
                      """Adam optimizer epsilon.""")

# Geometric layout.
LSTM_NUM_UNITS = 2048                # Number of hidden units per LSTM cell.
LSTM_NUM_LAYERS = 1                  # Number of stacked BDLSTM layers.
DENSE_NUM_UNITS = 2048               # Number of units per dense layer.


# Logging & Output
tf.flags.DEFINE_integer('max_steps', 1000000,
                        """Number of batches to run.""")
tf.flags.DEFINE_integer('log_frequency', 101,
                        """How often (every x steps) to log results to the console.""")

# Data set
tf.flags.DEFINE_integer('sampling_rate', 16000,
                        """The sampling rate of the audio files (2 * 8kHz).""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
tf.flags.DEFINE_string('train_dir', '/tmp/speech/train',
                       """Directory where to write event logs and checkpoints.""")
tf.flags.DEFINE_integer('num_examples_train', 3696,
                        """Number of examples in the training set.""")
tf.flags.DEFINE_integer('num_examples_test', 1344,
                        """Number of examples in the testing/evaluation set.""")

NUM_CLASSES = num_classes()

# Miscellaneous
TF_FLOAT = tf.float32   # ctc_xxxx functions don't support float64. See #13
NP_FLOAT = np.float32   # ctc_xxxx functions don't support float64. See #13


# Export names.
FLAGS = tf.flags.FLAGS


def get_parameters():
    """Generate a string containing the training parameters.

    Returns:
        (str): Training parameters.
    """
    ps = 'Learning Rage (lr={}, epochs={}, decay={}); BDLSTM (num_units={}, num_layers={}); ' \
         'Training (max_steps={}, log_frequency={})'
    return ps.format(FLAGS.learning_rate, FLAGS.num_epochs_per_decay,
                     FLAGS.learning_rate_decay_factor, LSTM_NUM_UNITS, LSTM_NUM_LAYERS,
                     FLAGS.max_steps, FLAGS.log_frequency)
