"""Collection of hyper parameters, network layout, and reporting options."""

from os import path
import tensorflow as tf
import numpy as np
from multiprocessing import cpu_count

from asr.labels import num_classes


# Constants describing the training process.
tf.flags.DEFINE_string('train_dir', '../speech_checkpoints/3d1c2d_global_full',
                       "Directory where to write event logs and checkpoints.")
tf.flags.DEFINE_integer('batch_size', 8,
                        "Number of samples within a batch.")


# Performance.
tf.flags.DEFINE_boolean('use_cudnn', True,
                        "Whether to use Nvidia cuDNN implementations or (False) the default "
                        "TensorFlow implementation.")
tf.flags.DEFINE_integer('num_threads', cpu_count(),
                        """Number of threads used to preload data.""")


# Learning Rate.
tf.flags.DEFINE_integer('max_epochs', 20,
                        "Number of epochs to run. [Deep Speech 1] uses about 20 epochs.")
tf.flags.DEFINE_float('learning_rate', 0.5e-4,
                      "Initial learning rate.")
tf.flags.DEFINE_float('learning_rate_decay_factor', 3/4,
                      "Learning rate decay factor.")
tf.flags.DEFINE_integer('steps_per_decay', 50000,
                        "Number of steps after which learning rate decays.")


# Adam Optimizer.
tf.flags.DEFINE_float('adam_beta1', 0.9,
                      "Adam optimizer beta_1 power.")
tf.flags.DEFINE_float('adam_beta2', 0.999,
                      "Adam optimizer beta_2 power.")
tf.flags.DEFINE_float('adam_epsilon', 1e-8,
                      "Adam optimizer epsilon.")


# CTC loss and decoder.
tf.flags.DEFINE_integer('beam_width', 1024,
                        "Beam width used in the CTC `beam_search_decoder`.")
tf.flags.DEFINE_boolean('use_warp_ctc', False,
                        "Weather to use Baidu's `warp_ctc_loss` or TensorFlow's `ctc_loss`.")


# Dropout.
tf.flags.DEFINE_float('conv_dropout_rate', 0.0,
                      "Dropout rate for convolutional layers.")
tf.flags.DEFINE_float('rnn_dropout_rate', 0.0,
                      "Dropout rate for the RNN cell layers.")
tf.flags.DEFINE_float('dense_dropout_rate', 0.1,
                      "Dropout rate for dense layers.")


# Layer and activation options.
tf.flags.DEFINE_string('used_model', 'ds1',
                       ("Used inference model. Supported are 'ds1', and 'ds2'. "
                        "Also see `FLAGS.feature_drop_every_second_frame`."))

tf.flags.DEFINE_multi_integer('conv_filters', [32, 32, 96],
                              "Number of filters for each convolutional layer.")

tf.flags.DEFINE_integer('num_layers_rnn', 1,
                        "Number of stacked RNN cells.")
tf.flags.DEFINE_integer('num_units_rnn', 2048,
                        "Number of hidden units in each of the RNN cells.")

tf.flags.DEFINE_integer('num_units_dense', 2048,
                        "Number of units per dense layer.")

tf.flags.DEFINE_float('relu_cutoff', 20.0,
                      "Cutoff ReLU activations that exceed the cutoff.")


# Input features.
tf.flags.DEFINE_string('feature_type', 'mel',
                       "Type of input features. Supported types are: 'mel' and 'mfcc'.")
tf.flags.DEFINE_string('feature_normalization', 'local',
                       ("Type of normalization applied to input features."
                        "Supported are: 'none', 'global', 'local', and 'local_scalar'"))
tf.flags.DEFINE_boolean('features_drop_every_second_frame', False,
                        "[Deep Speech 1] like dropping of every 2nd input time frame.")


# Dataset.
tf.flags.DEFINE_integer('sampling_rate', 16000,
                        "The sampling rate of the audio files (2 * 8kHz).")
tf.flags.DEFINE_integer('num_examples_train', 570854,
                        "Number of examples in the training set. `test.txt`")
tf.flags.DEFINE_integer('num_examples_test', 3556,
                        "Number of examples in the testing/evaluation set. `test.txt`")
tf.flags.DEFINE_integer('num_examples_dev', 2590,
                        "Number of examples in the validation set. `dev.txt`")
tf.flags.DEFINE_integer('num_classes', num_classes(),
                        "Number of classes. Contains the additional CTC <blank> label.")


# Logging.
tf.flags.DEFINE_integer('log_frequency', 500,
                        "How often (every `log_frequency` steps) to log results.")
tf.flags.DEFINE_integer('num_samples_to_report', 4,
                        "The maximum number of decoded and original text samples to report.")


# Miscellaneous.
tf.flags.DEFINE_bool('delete', False,
                     "Whether to delete old checkpoints, or resume training.")
tf.flags.DEFINE_integer('random_seed', 0,
                        "TensorFlow random seed. Set to `0` to use the current timestamp instead.")
tf.flags.DEFINE_boolean('log_device_placement', False,
                        "Whether to log device placement.")
tf.flags.DEFINE_boolean('allow_vram_growth', True,
                        "Allow TensorFlow to allocate VRAM as needed, as opposed to allocating the "
                        "whole VRAM at program start.")


# Export names.
TF_FLOAT = tf.float32   # ctc_*** functions don't support float64. See #13
NP_FLOAT = np.float32   # ctc_*** functions don't support float64. See #13

BASE_PATH = path.split(path.dirname(path.realpath(__file__)))[0]    # Path to git root.

MIN_EXAMPLE_LENGTH = 0.7    # Minimum length of examples in datasets (in seconds).
MAX_EXAMPLE_LENGTH = 17.0   # Maximum length of examples in datasets (in seconds).

FLAGS = tf.flags.FLAGS


def get_parameters():
    """Generate a summary containing the training, and network parameters.

    Returns:
        (str): Summary of training parameters.
    """
    s = '\nLearning Rate (lr={}, steps_per_decay={:,d}, decay_factor={});\n' \
        'GPU-Options (use_warp_ctc={}; use_cudnn={});\n' \
        'Model (used_model={}, beam_width={:,d})\n' \
        'Conv (conv_filters={}); Dense (num_units={:,d});\n' \
        'RNN (num_units={:,d}, num_layers={:,d});\n' \
        'Training (batch_size={:,d}, max_epochs={:,d} ({:,d} steps; ' \
        '{:,d} steps_per_epoch), log_frequency={:,d});\n' \
        'Features (type={}, normalization={}, skip_every_2nd_frame={});'

    s = s.format(FLAGS.learning_rate, FLAGS.steps_per_decay, FLAGS.learning_rate_decay_factor,
                 FLAGS.use_warp_ctc, FLAGS.use_cudnn,
                 FLAGS.used_model, FLAGS.beam_width,
                 FLAGS.conv_filters, FLAGS.num_units_dense,
                 FLAGS.num_units_rnn, FLAGS.num_layers_rnn,
                 FLAGS.batch_size, FLAGS.max_epochs,
                 FLAGS.max_epochs * FLAGS.num_examples_train // FLAGS.batch_size,
                 FLAGS.num_examples_train // FLAGS.batch_size, FLAGS.log_frequency,
                 FLAGS.feature_type, FLAGS.feature_normalization,
                 FLAGS.features_drop_every_second_frame)

    return s
