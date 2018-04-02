"""Routines to load the `TIMIT`_ corpus and and
pre process the audio files and labels.

.. _TIMIT:
    https://vcs.zwuenf.org/agct_data/timit
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow.contrib as tfc

from s_utils import LabelManager


NUM_MFCC = 13
NUM_EXAMPLES_PER_EPOCH_TRAIN = 4620
NUM_EXAMPLES_PER_EPOCH_EVAL = 1680
NUM_CLASSES = LabelManager().num_classes()
DATA_PATH = '/home/marc/workspace/speech/data'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('sampling_rate', 16000,
                            """The sampling rate of the audio files (2 * 8kHz).""")


def inputs_train(batch_size):
    """Construct input for speech training.

    Args:
        batch_size (int):
            (Maximum) number of samples per batch.
            See: _generate_batch() and `allow_smaller_final_batch=True`

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
    # Info: Longest label list in TIMIT train/test is 79 characters long.
    train_txt_path = os.path.join(DATA_PATH, 'train.txt')
    sample_list, label_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.33
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_TRAIN * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + 3 * batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue = tf.train.slice_input_producer(
            [file_names, labels], capacity=capacity, num_epochs=None, shuffle=False)

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)

        # Read the sample from disk and extract it's features.
        sample, sample_len = tf.py_func(_load_sample, [sample_queue], [tf.float32, tf.int32])

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([None, NUM_MFCC])
        sample_len.set_shape([])    # Shape for scalar is [].

        print('Generating training batches of size {}. Queue capacity is {}. '
              .format(batch_size, capacity))

        sequences, seq_length, labels = _generate_batch(sample, sample_len, label_queue,
                                                        batch_size, capacity)
        return sequences, seq_length, labels


def inputs():
    # TODO: Rewrite this function to match inputs_train().
    raise NotImplementedError


def _load_sample(file_path):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

    Returns:
        np.ndarray:
            2D array with [time, NUM_MFCC] shape, containing float32.
        np.ndarray:
            1D array, containing int32.

    Review:
        * Review NUM_MFCC = 13.
        * Review if (mfcc + mfcc_delta) are better features than pure mfcc.
    """
    file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    # y, sr = rosa.load(file_path, sr=None, mono=True)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # Set generally used variables.
    # At 22050 Hz, 512 samples ~= 23ms. At 16000 Hz, 512 samples = 32ms.
    hop_length = 200    # Number of samples between successive frames e.g. columns if a spectrogram.
    f_max = sr / 2.     # Maximum frequency (Nyquist rate).
    f_min = 64.         # Minimum frequency.
    n_fft = 1024        # Number of samples in a frame.
    n_mfcc = NUM_MFCC   # Number of Mel cepstral coefficients to extract.

    db_pow = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=400)) ** 2

    s_mel = librosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                           fmax=f_max, fmin=f_min, n_mels=80)

    s_mel = librosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = librosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    # And the first-order differences (delta features).
    # mfcc_delta = rosa.feature.delta(mfcc, width=5, order=1)

    sample = mfcc.astype(np.float32)
    sample = np.swapaxes(sample, 0, 1)
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    return sample, sample_len


def _read_file_list(path, label_manager=LabelManager()):
    """Generate synchronous lists of all samples with their respective lengths and labels.
    Labels are converted from characters to integers. See: `s_utils.LabelManager`.

    Args:
        path (str):
            Path to the training or testing .TXT files, e.g. `/some/path/timit/train.txt`
        label_manager (s_utils.LabelManager):
            Character to integer mapping.

    Returns:
        [str]: List of absolute paths to .WAV files.
        [[int]]: List of labels filtered and converted to integer lists.
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

        return sample_paths, labels


def _generate_batch(sequence, seq_len, label, batch_size, capacity):
    """Construct a queued batch of sample sequences and labels.

    Args:
        sequence (tf.Tensor):
            2D tensor of shape [time, NUM_MFCC] with type float32.
        seq_len (tf.Tensor):
            1D tensor of shape [1] with type int32.
        label (tf.Tensor):
            1D tensor of shape [<length label>] with type int32.
        batch_size (int):
            (Maximum) number of samples per batch.
        capacity (int):
            The maximum number of minibatches in the top queue,
            and also the maximum number of elements within each bucket.

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
    num_pre_process_threads = 12

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    seq_length, (sequences, labels) = tfc.training.bucket_by_sequence_length(
        input_length=seq_len,
        tensors=[sequence, label],
        batch_size=batch_size,
        bucket_boundaries=[160, 200, 220, 240, 300],  # L8ER Find good bucket sizes.
        num_threads=num_pre_process_threads,
        capacity=capacity,
        # Pads smaller batch elements (sequence and label) to the size of the longest one.
        dynamic_pad=True,
        allow_smaller_final_batch=False             # review Test if it works? Return batch_size
    )

    # Display the training images in the visualizer.
    batch_size_t = tf.shape(sequences)[0]
    summary_batch = tf.reshape(sequences, [batch_size_t, -1, NUM_MFCC, 1])
    tf.summary.image('sample', summary_batch, max_outputs=batch_size)
    tf.summary.histogram('labels_hist', labels)

    return sequences, seq_length, labels
