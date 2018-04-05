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

from s_params import FLAGS, NP_FLOAT, TF_FLOAT
import s_labels


NUM_MFCC = 13
INPUT_LENGTH = NUM_MFCC * 2
DATA_PATH = '/home/marc/workspace/speech/data'


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
        tf.Tensor:
            2D Tensor with the original strings.
    """
    # Info: Longest label list in TIMIT train/test is 79 characters long.
    train_txt_path = os.path.join(DATA_PATH, 'train.txt')
    sample_list, label_list, original_list = _read_file_list(train_txt_path)

    with tf.name_scope('train_input'):
        # Convert lists to tensors.
        file_names = tf.convert_to_tensor(sample_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        originals = tf.convert_to_tensor(original_list, dtype=tf.string)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.2
        min_queue_examples = int(FLAGS.num_examples_train * min_fraction_of_examples_in_queue)
        capacity = min_queue_examples + 3 * batch_size

        # Create an input queue that produces the file names to read.
        sample_queue, label_queue, originals_queue = tf.train.slice_input_producer(
            [file_names, labels, originals], capacity=capacity, num_epochs=None, shuffle=True)

        # Reinterpret the bytes of a string as a vector of numbers.
        label_queue = tf.decode_raw(label_queue, tf.int32)

        # Read the sample from disk and extract it's features.
        sample, sample_len = tf.py_func(_load_sample, [sample_queue], [TF_FLOAT, tf.int32])

        # Restore shape, since `py_func` forgets it.
        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        sample.set_shape([None, INPUT_LENGTH])
        sample_len.set_shape([])    # Shape for scalar is [].

        print('Generating training batches of size {}. Queue capacity is {}. '
              .format(batch_size, capacity))

        sequences, seq_length, labels, originals = _generate_batch(
            sample, sample_len, label_queue, originals_queue, batch_size, capacity)

        # Reshape labels for CTC loss.
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/dense_to_sparse
        labels = tfc.layers.dense_to_sparse(labels)

        return sequences, seq_length, labels, originals


def inputs(batch_size):
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
            2D array with [time, num_features] shape, containing float.
        np.ndarray:
            1 element array, containing a single int32.

    Review:
        * (mfcc + mfcc_delta) better features than pure mfcc?
        * Normalize mfcc_delta.
    """
    file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # Set generally used variables.
    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    hop_length = 200    # Number of samples between successive frames e.g. columns if a spectrogram.
    f_max = sr / 2.     # Maximum frequency (Nyquist rate).
    f_min = 64.         # Minimum frequency.
    n_fft = 1024        # Number of samples in a frame.
    n_mfcc = NUM_MFCC   # Number of Mel cepstral coefficients to extract.
    n_mels = 80         # Number of Mel bins to generate
    win_length = 333    # Window length

    db_pow = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length)) ** 2

    s_mel = librosa.feature.melspectrogram(S=db_pow, sr=sr, hop_length=hop_length,
                                           fmax=f_max, fmin=f_min, n_mels=n_mels)

    s_mel = librosa.power_to_db(s_mel, ref=np.max)

    # Compute MFCC features from the mel spectrogram.
    mfcc = librosa.feature.mfcc(S=s_mel, sr=sr, n_mfcc=n_mfcc)

    # And the first-order differences (delta features).
    mfcc_delta = librosa.feature.delta(mfcc, width=5, order=1)

    # Combine MFCC with MFCC_delta
    sample = np.concatenate([mfcc, mfcc_delta], axis=0)

    sample = sample.astype(NP_FLOAT)
    sample = np.swapaxes(sample, 0, 1)
    sample_len = np.array(sample.shape[0], dtype=np.int32)
    sample = (sample - np.mean(sample)) / np.std(sample)    # review useful? also try norm.

    # `sample`: [time, num_features], `sample_len`: scalar
    return sample, sample_len


def _read_file_list(path):
    """Generate synchronous lists of all samples with their respective lengths and labels.
    Labels are converted from characters to integers.
    See: `s_labels`.

    Args:
        path (str):
            Path to the training or testing .TXT files, e.g. `/some/path/timit/train.txt`

    Returns:
        [str]: List of absolute paths to .WAV files.
        [[int]]: List of labels filtered and converted to integer lists.
        [str]: List of original strings.
    """
    with open(path) as f:
        lines = f.readlines()

        sample_paths = []
        labels = []
        originals = []
        tmp = 0     # TODO remove
        for line in lines:
            sample_path, label = line.split(' ', 1)
            sample_paths.append(os.path.join(DATA_PATH, 'timit/TIMIT', sample_path))
            label = label.strip()
            originals.append(label)
            label = [s_labels.ctoi(c) for c in label]
            label = np.array(label, dtype=np.int32).tostring()
            labels.append(label)

            # TODO remove
            # tmp += 1
            # if tmp >= 8:
            #     break

        return sample_paths, labels, originals


def _generate_batch(sequence, seq_len, label, original, batch_size, capacity):
    """Construct a queued batch of sample sequences and labels.

    Args:
        sequence (tf.Tensor):
            2D tensor of shape [time, INPUT_LENGTH] with type float.
        seq_len (tf.Tensor):
            1D tensor of shape [1] with type int32.
        label (tf.Tensor):
            1D tensor of shape [<length label>] with type int32.
        original (tf.Tensor):
            1D tensor of shape [<length original text>] with type tf.string.
            The original text.
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
        tf.Tensor:
            2D Tensor with the original strings.
    """
    num_pre_process_threads = 6
    bucket_boundaries = [130, 170, 200, 230, 270, 330]   # L8ER Find good bucket sizes.

    # https://www.tensorflow.org/api_docs/python/tf/contrib/training/bucket_by_sequence_length
    seq_length, (sequences, labels, originals) = tfc.training.bucket_by_sequence_length(
        input_length=seq_len,
        tensors=[sequence, label, original],
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        num_threads=num_pre_process_threads,
        capacity=capacity // (len(bucket_boundaries) + 2),
        # Pads smaller batch elements (sequence and label) to the size of the longest one.
        dynamic_pad=True,
        allow_smaller_final_batch=False             # review Test if it works? Return batch_size
    )

    # Display the training images in the visualizer.
    batch_size_t = tf.shape(sequences)[0]
    summary_batch = tf.reshape(sequences, [batch_size_t, -1, INPUT_LENGTH, 1])
    tf.summary.image('sample', summary_batch, max_outputs=batch_size)
    tf.summary.histogram('labels', labels)

    return sequences, seq_length, labels, originals
