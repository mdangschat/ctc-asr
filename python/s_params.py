"""Collection of hyper parameters."""


from tensorflow import flags

from s_labels import num_classes


# Constants describing the training process.
flags.DEFINE_integer('batch_size', 4,
                     """(Maximum) Number of samples within a batch.""")

NUM_EPOCHS_PER_DECAY = 1.0          # Number of epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.50   # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001      # Initial learning rate.


# Logging & Output
flags.DEFINE_integer('max_steps', 50000,
                     """Number of batches to run.""")
flags.DEFINE_integer('log_frequency', 47,
                     """How often (every x steps) to log results to the console.""")

# Miscellaneous & Data set
flags.DEFINE_integer('sampling_rate', 16000,
                     """The sampling rate of the audio files (2 * 8kHz).""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_string('train_dir', '/tmp/s_train',
                    """Directory where to write event logs and checkpoints.""")

NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
NUM_CLASSES = num_classes()


# Export names.
FLAGS = flags.FLAGS
