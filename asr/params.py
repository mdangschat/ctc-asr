"""Collection of hyper parameters, network layout, and reporting options."""

import math

import tensorflow as tf
import numpy as np

from asr.s_labels import num_classes


# Constants describing the training process.
tf.flags.DEFINE_string('train_dir', '/home/marc/workspace/speech_checkpoints/3c3r2d_1',
                       """Directory where to write event logs and checkpoints.""")
tf.flags.DEFINE_integer('batch_size', 8,
                        """Number of samples within a batch.""")

# Performance.
tf.flags.DEFINE_bool('use_cudnn', True,
                     """Whether to use Nvidia cuDNN implementations or (False) the default 
                     TensorFlow implementation.""")
tf.flags.DEFINE_integer('num_threads', 8,
                        """Number of threads used to preload data.""")

# Learning Rate.
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      """Initial learning rate.""")
tf.flags.DEFINE_float('learning_rate_decay_factor', 3/5,
                      """Learning rate decay factor.""")
tf.flags.DEFINE_integer('steps_per_decay', 75000,
                        """Number of steps after which learning rate decays.""")

# Adam Optimizer.
tf.flags.DEFINE_float('adam_beta1', 0.9,
                      """Adam optimizer beta_1 power.""")
tf.flags.DEFINE_float('adam_beta2', 0.999,
                      """Adam optimizer beta_2 power.""")
tf.flags.DEFINE_float('adam_epsilon', 1e-8,
                      """Adam optimizer epsilon.""")

# CTC loss and decoder.
tf.flags.DEFINE_integer('beam_width', 1024,
                        """Beam width used in the CTC `beam_search_decoder`.""")
tf.flags.DEFINE_bool('use_warp_ctc', False,
                     """Weather to use Baidu's `warp_ctc_loss` or TensorFlow's `ctc_loss`.""")

# Dropout.
tf.flags.DEFINE_float('conv_dropout_rate', 0.0,
                      """Dropout rate for convolutional layers.""")
tf.flags.DEFINE_float('rnn_dropout_rate', 0.0,
                      """Dropout rate for the RNN cell layers.""")
tf.flags.DEFINE_float('dense_dropout_rate', 0.1,
                      """Dropout rate for dense layers.""")

# Layer and activation options.
tf.flags.DEFINE_multi_integer('conv_filters', [32, 32, 96],
                              """Number of filters for each convolutional layer.""")

tf.flags.DEFINE_integer('num_layers_rnn', 3,
                        """Number of stacked RNN cells.""")
tf.flags.DEFINE_integer('num_units_rnn', 2048,
                        """Number of hidden units in each of the RNN cells.""")

tf.flags.DEFINE_integer('num_units_dense', 2048,
                        """Number of units per dense layer.""")

tf.flags.DEFINE_float('relu_cutoff', 20.0,
                      """Cutoff ReLU activations that exceed the cutoff.""")

# Logging and Output.
tf.flags.DEFINE_integer('max_epochs', 20,
                        """Number of epochs to run. [Deep Speech 1] uses 15 to 20 epochs.""")
tf.flags.DEFINE_integer('log_frequency', 250,
                        """How often (every `log_frequency` steps) to log results.""")
tf.flags.DEFINE_integer('num_samples_to_report', 4,
                        """The maximum number of decoded and original text samples to report.""")

# Input features.
tf.flags.DEFINE_string('feature_type', 'mel',
                       """Type of input features. Supported types are: 'mel' and 'mfcc'.""")
tf.flags.DEFINE_string('feature_normalization', 'local',
                       """Type of normalization applied to input features. 
                       Supported are: 'none', 'global', 'local', and 'local_scalar'""")

# Dataset.
tf.flags.DEFINE_integer('sampling_rate', 16000,
                        """The sampling rate of the audio files (2 * 8kHz).""")
tf.flags.DEFINE_integer('num_examples_train', 272537,
                        """Number of examples in the training set. `test.txt`""")
tf.flags.DEFINE_integer('num_examples_test', 3679,
                        """Number of examples in the testing/evaluation set. `test.txt`""")
tf.flags.DEFINE_integer('num_examples_dev', 2703,
                        """Number of examples in the validation set. `dev.txt`""")
tf.flags.DEFINE_integer('num_classes', num_classes(),
                        """Number of classes. Contains the additional CTC <blank> label.""")

# Evaluation.
tf.flags.DEFINE_string('eval_dir', '',
                       """If set, evaluation log data will be stored here, instead of the default
                       directory `f'{FLAGS.train_dir}_eval'.""")

# Miscellaneous.
tf.flags.DEFINE_bool('delete', False,
                     """Whether to delete old checkpoints, or resume training.""")
tf.flags.DEFINE_integer('random_seed', 4711,
                        """TensorFlow random seed.""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        """Whether to log device placement.""")
tf.flags.DEFINE_boolean('allow_vram_growth', True,
                        """Allow TensorFlow to allocate VRAM as needed, 
                        as opposed to allocating the whole VRAM at program start.""")


# Export names.
TF_FLOAT = tf.float32   # ctc_*** functions don't support float64. See #13
NP_FLOAT = np.float32   # ctc_*** functions don't support float64. See #13

FLAGS = tf.flags.FLAGS


def get_parameters():
    """Generate a string containing the training, and network parameters.

    Returns:
        (str): Training parameters.
    """
    s = '\nLearning Rate (lr={}, steps_per_decay={:,d}, decay_factor={});\n' \
        'GPU-Options (use_warp_ctc={}; use_cudnn={});\n' \
        'Conv (conv_filters={});\n' \
        'RNN (num_units={:,d}, num_layers={:,d});\n' \
        'Dense (num_units={:,d});\n' \
        'Decoding (beam_width={:,d});\n' \
        'Training (batch_size={:,d}, max_epochs={:,d} ({:,d} steps; ' \
        '{:,d} steps_per_epoch), log_frequency={:,d});'
    return s.format(FLAGS.learning_rate, FLAGS.steps_per_decay, FLAGS.learning_rate_decay_factor,
                    FLAGS.use_warp_ctc, FLAGS.use_cudnn,
                    FLAGS.conv_filters,
                    FLAGS.num_units_rnn, FLAGS.num_layers_rnn,
                    FLAGS.num_units_dense,
                    FLAGS.beam_width,
                    FLAGS.batch_size, FLAGS.max_epochs,
                    math.floor(FLAGS.max_epochs * FLAGS.num_examples_train / FLAGS.batch_size),
                    int(FLAGS.num_examples_train // FLAGS.batch_size), FLAGS.log_frequency)
