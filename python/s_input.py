"""L8ER Documentation
Routines to load the traffic sign (TS) corpus [BelgiumTS (cropped images)]
and transform the images into an usable format.

.. _BelgiumTS:
   http://btsd.ethz.ch/shareddata/

"""

import os
import numpy as np
import librosa
import tensorflow as tf

from loader import load_input
import s_utils


NUMBER_CLASSES = 26     # review
MAX_INPUT_LEN = 80      # review needed?
SAMPLING_RATE = 16000
NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
DATA_PATH = '/home/marc/workspace/speech/data'

FLAGS = tf.app.flags.FLAGS


def inputs_train(data_dir, batch_size):
    """Construct input for TS training.
    review Documentation

    Args:
        data_dir: Path to the data directory.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of
                [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]] size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    train_txt_path = os.path.join(data_dir, 'train.txt')
    # Longest label list in train/test is 79 characters.
    sample_list, label_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        print('train_input:', file_names, labels)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_TRAIN * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + 3 * batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue = tf.train.slice_input_producer(
            [file_names, labels], capacity=capacity, num_epochs=None, shuffle=False)

        print('queues:', sample_queue, label_queue)

        # review: The body of the function (i.e. func) will not be serialized in a GraphDef.
        # py_func: You should not use this function if you need to serialize your model
        # and restore it in a different environment.
        sample = tf.py_func(_read_sample, [sample_queue], tf.float32)
        label = label_queue
        label = label[10]   # TODO: Remove this, this is only for testing!
        print('py_func:', sample, type(sample), label)

        # Restore shape. See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([100, 13])    # review both shapes
        print('reset shape:', sample, label)

        print('Filling the queue with {} images before starting to train. '
              'Queue capacity is {}. '
              'This will take a few minutes.'
              .format(min_queue_examples, capacity))

    return _generate_batch(sample, label, min_queue_examples, batch_size, shuffle=True)


def inputs():
    # TODO: Rewrite this function to match inputs_train().
    """Construct fitting input for the evaluation process.

    Args:
        eval_data (boolean): Indicating if one should use the train or eval data set.
        data_dir (str): Path to the data directory.
        image_shape ([int]): Three element shape array.
                     E.g. [32, 32, 3] for colored images and [32, 32, 1] for monochrome images.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]]
                size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    raise NotImplementedError


def _read_sample(sample_queue, expected_sr=None):
    """Reads the wave file and converts it into an MFCC.
    review Documentation

    Args:
        sample_queue: A TensorFlow queue of tuples with the file names to read from and labels.
                      Compare: tf.train.slice_input_producer

    Returns:
        reshaped_image: A single example.
        label: The corresponding label.
    """
    file_path = str(sample_queue, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Set generally used variables. TODO: Document their purpose.
    # At 22050 Hz, 512 samples ~= 23ms. At 16000 Hz, 512 samples ~= TODO ms.
    hop_length = 200
    f_max = sr / 2.
    f_min = 64.
    n_mfcc = 13

    if expected_sr is not None:
        if not sr == expected_sr:
            raise ValueError('Sample rate of {:,d} does not match the required rate of {:,d}.'
                             .format(sr, expected_sr))

    db_pow = np.abs(librosa.stft(y=y, n_fft=1024, hop_length=hop_length, win_length=400)) ** 2

    s_mel = librosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                           fmax=f_max, fmin=f_min, n_mels=80)

    s_mel = librosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = librosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    # And the first-order differences (delta features).
    # mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    return mfcc


def _read_file_list(path, label_manager=s_utils.LabelManager()):
    """Generate two synchronous lists of all image samples and their respective labels
    within the provided path.
    review Documentation
    review: Labels are converted from characters to integers. See: Labels.

    Args:
        path (str): Path to the training or testing folder of the TS data set.

    Returns:
        file_names ([str]): A list of file name strings.
        labels ([[int]]): A list of labels as integer.

    .. _StackOverflow:
       https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    """
    print('Opening: {}'.format(path))     # todo remove
    with open(path) as f:
        lines = f.readlines()

        samples = []
        labels = []
        for line in lines:
            sample, label = line.split(' ', 1)
            samples.append(os.path.join(DATA_PATH, 'timit/TIMIT', sample))
            label = [label_manager.ctoi(c) for c in label.strip()]
            pad_len = MAX_INPUT_LEN - len(label)
            labels.append(np.pad(np.array(label, dtype=np.int32), (0, pad_len), 'constant'))

        return samples, np.array(labels)


def _generate_batch(sample, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    review Documentation

    Args:
        sample: 3D tensor of [height, width, 1] of type float32.
        label: 1D tensor of type int32.
        min_queue_examples (int): Minimum number of samples to retain in the queue that
                                  provides the example batches.
        batch_size (int): Number of images per batch.
        shuffle (boolean): Indicating whether to use shuffling queue or not.

    Returns:
        images: Images 4D tensor of [batch_size, height, width, 1] size.
        labels: Labels 1D tensor of [batch_size] size.
    """
    num_pre_process_threads = 12
    capacity = min_queue_examples + 3 * batch_size

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [sample, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [sample, label],
            batch_size=batch_size,
            num_threads=num_pre_process_threads,
            capacity=capacity
        )

    # Display the training images in the visualizer.
    # tf.summary.image('images', image_batch, max_outputs=10)    # L8ER: Summay options for audio?

    return image_batch, label_batch
