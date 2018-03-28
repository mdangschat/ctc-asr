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
import tensorflow.contrib as tfc

import s_utils


NUMBER_CLASSES = 26         # review
MAX_LABEL_LEN = 80          # review
MAX_INPUT_LEN = 666         # review
NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
DATA_PATH = '/home/marc/workspace/speech/data'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('sampling_rate', 16000,
                            """The sampling rate of the audio files (2 * Hz).""")


def inputs_train(data_dir, batch_size):
    """Construct input for speech training.
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
    sample_list, label_list, label_len_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        label_lens = tf.convert_to_tensor(label_len_list, dtype=tf.int32)
        print('train_input:', file_names, labels, label_lens)

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
        # label = label_queue       # TODO Reactivate
        label = np.array(1, dtype=np.int32)   # TODO: Remove this, this is only for testing!
        label_len = np.array(1, dtype=np.int32)
        print('py_func:', sample, sample.shape, label)

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([None, 13])    # review shape, use []
        print('set_shape:', sample, sample.shape)

        print('Generating training batches. This may take some time.')
        return _generate_batch(sample, label, label_len, batch_size)


def inputs():
    # L8ER: Rewrite this function to match inputs_train().
    raise NotImplementedError


def _read_sample(sample_queue):
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

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # Set generally used variables. TODO: Document their purpose.
    # At 22050 Hz, 512 samples ~= 23ms. At 16000 Hz, 512 samples ~= TODO ms.
    hop_length = 200
    f_max = sr / 2.
    f_min = 64.
    n_mfcc = 13

    db_pow = np.abs(librosa.stft(y=y, n_fft=1024, hop_length=hop_length, win_length=400)) ** 2

    s_mel = librosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                           fmax=f_max, fmin=f_min, n_mels=80)

    s_mel = librosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = librosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    # And the first-order differences (delta features).
    # mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    # TODO Remove prints
    sample = mfcc.astype(np.float32)
    assert sample.shape[1] <= MAX_INPUT_LEN, 'MAX_INPUT_LEN to low: %d' % sample.shape[1]
    # print('sample:', sample.shape)
    # sample = np.pad(sample, [[0, 0], [0, MAX_INPUT_LEN - sample.shape[1]]], 'constant')
    # print('sample pad:', sample.shape)
    sample = np.swapaxes(sample, 0, 1)
    # print('sample pad swap:', sample.shape)
    return sample


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
    with open(path) as f:
        lines = f.readlines()

        samples = []
        labels = []
        label_lens = []
        for line in lines:
            sample, label = line.split(' ', 1)
            samples.append(os.path.join(DATA_PATH, 'timit/TIMIT', sample))
            label = [label_manager.ctoi(c) for c in label.strip()]
            pad_len = MAX_LABEL_LEN - len(label)
            label = np.pad(np.array(label, dtype=np.int32), [0, pad_len], 'constant')
            labels.append(label)
            label_lens.append(len(label))

        return samples, np.array(labels), np.array(label_lens)


def _generate_batch(sample, label, label_len, batch_size):
    """Construct a queued batch of images and labels.
    review Documentation

    Args:
        sample: 3D tensor of [height, width, 1] of type float32.
        label: 1D tensor of type int32.
        label_len: 1D tensor of type int32.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images 4D tensor of [batch_size, height, width, 1] size.
        labels: Labels 1D tensor of [batch_size] size.
    """
    num_pre_process_threads = 12
    capacity = 10 + 3 * batch_size

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    batch_sequence_length, (sample_batch, label_batch) = tfc.training.bucket_by_sequence_length(
        input_length=label_len,
        tensors=[sample, label],
        batch_size=batch_size,
        bucket_boundaries=[l for l in range(10, MAX_INPUT_LEN)],
        num_threads=num_pre_process_threads,
        capacity=capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=False
    )

    # Display the training images in the visualizer.
    # summary_batch = tf.reshape(sample, [1, None, 13, 1])     # L8ER: Use correct shape.
    # tf.summary.image('input_data', summary_batch, max_outputs=1)

    return sample_batch, batch_sequence_length, label_batch
