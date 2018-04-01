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


_label_manager = s_utils.LabelManager()
NUM_CLASSES = _label_manager.num_classes()   # review
MAX_INPUT_LEN = 666         # review
INPUT_PAD_LEN = 8           # review
NUM_MFCC = 13
NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
DATA_PATH = '/home/marc/workspace/speech/data'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('sampling_rate', 16000,
                            """The sampling rate of the audio files (2 * 8kHz).""")


def inputs_train(batch_size):
    """Construct input for speech training.
    review Documentation

    Args:
        batch_size (int): Number of images per batch.

    Returns:
        images: Images a 4D tensor of
                [batch_size, IMAGE_SHAPE[0], IMAGE_SHAPE[1], INPUT_SHAPE[2]] size.
        labels: Labels a 1D tensor of [batch_size] size.
    """
    # Info: Longest label list in TIMIT train/test is 79 characters long.
    train_txt_path = os.path.join(DATA_PATH, 'train.txt')
    sample_list, label_list, label_len_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        label_lens = tf.convert_to_tensor(label_len_list, dtype=tf.int32)
        print('train_input:', file_names, labels, label_lens)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_TRAIN * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + 3 * batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue, label_len_queue = tf.train.slice_input_producer(
            [file_names, labels, label_lens], capacity=capacity, num_epochs=None, shuffle=False)
        print('queues:', sample_queue, label_queue, label_len_queue)

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)
        print('label_queue decode_raw:', label_queue)

        # Read the sample from disk and extract it's features.
        sample, sample_len = tf.py_func(_read_sample, [sample_queue], [tf.float32, tf.int32])
        print('py_func:', sample, sample_len, labels, label_len_queue)

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([None, NUM_MFCC])
        sample_len.set_shape([])    # Shape for scalar is [].
        print('set_shape:', sample, sample.shape, sample_len, sample_len.shape)

        print('Generating training batches. This may take some time.')
        return _generate_batch(sample, label_queue, sample_len, batch_size, capacity)


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
    n_mfcc = NUM_MFCC

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

    sample = np.swapaxes(sample, 0, 1)
    sample_len = np.array(sample.shape[0], dtype=np.int32)      # TODO Not sure if np.array is needed here.
    print('sample:', sample.shape, sample_len)
    return sample, sample_len


def _read_file_list(path, label_manager=s_utils.LabelManager()):
    """Generate two synchronous lists of all image samples and their respective labels
    within the provided path.
    review Documentation
    review: Labels are converted from characters to integers. See: `s_utils.Labels`.

    Args:
        path (str): Path to the training or testing folder of the TS data set.
        label_manager (s_utils.LabelManager):

    Returns:
        file_names ([str]): A list of file name strings.
        labels ([[int]]): A list of labels as integer.

    .. _StackOverflow:
       https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    """
    with open(path) as f:
        lines = f.readlines()

        sample_paths = []
        labels = []
        label_lens = []
        for line in lines:
            sample_path, label = line.split(' ', 1)
            sample_paths.append(os.path.join(DATA_PATH, 'timit/TIMIT', sample_path))
            label = [label_manager.ctoi(c) for c in label.strip()]
            label_lens.append(len(label))
            label = np.array(label, dtype=np.int32).tostring()
            labels.append(label)

        return sample_paths, labels, label_lens


def _generate_batch(sequence, label, seq_len, batch_size, capacity):
    """Construct a queued batch of images and labels.
    review Documentation

    Args:
        sequence (): 3D tensor of [height, width, 1] of type float32.
        seq_len (): 1D tensor of type int32.
        label (): 1D tensor of type int32.
        batch_size (int): Number of images per batch.

    Returns:
        images: Images 4D tensor of [batch_size, height, width, 1] size.
        labels: Labels 1D tensor of [batch_size] size.
    """
    num_pre_process_threads = 1     # 12

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    sequence_length, (sample_batch, label_batch) = tfc.training.bucket_by_sequence_length(
        input_length=seq_len,
        tensors=[sequence, label],
        batch_size=batch_size,
        bucket_boundaries=[50, 100, 150, 200, 250],
        # bucket_boundaries=[l for l in         TODO test above
        #                    range(4 * INPUT_PAD_LEN, 50 * INPUT_PAD_LEN + 1, INPUT_PAD_LEN)],
        num_threads=num_pre_process_threads,
        capacity=capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=True     # review Test if it works?
    )

    # Display the training images in the visualizer.    TODO re-enable
    batch_size_t = tf.shape(sample_batch)[0]
    summary_batch = tf.reshape(sample_batch, [batch_size_t, -1, NUM_MFCC, 1])
    tf.summary.image('sample', summary_batch, max_outputs=batch_size)
    tf.summary.histogram('labels_hist', label_batch)

    # sequence_length = tf.Print(sequence_length, [tf.shape(sequence_length), sequence_length],
    #                            message='Batch_sequence_length:')
    # label_batch = tf.Print(label_batch, [tf.shape(label_batch)], message='Label_batch:')
    # sample_batch = tf.Print(sample_batch, [tf.shape(sample_batch)], message='Sample_batch:')
    print('batch_sequence_length:', sequence_length)
    return sample_batch, label_batch, sequence_length
