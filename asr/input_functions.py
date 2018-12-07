"""
Routines to load a corpus and perform the necessary pre processing on the audio files and labels.
Contains helper methods to load audio files, too.
"""

import csv
import os
import random

import numpy as np
import python_speech_features as psf
import tensorflow as tf
from scipy.io import wavfile

from asr.dataset.config import CORPUS_DIR
from asr.dataset.config import CSV_DELIMITER, CSV_FIELDNAMES, CSV_HEADER_LABEL, CSV_HEADER_PATH
from asr.labels import ctoi
from asr.params import BOUNDARIES, NP_FLOAT, FLAGS

WIN_LENGTH = 0.025  # Window length in seconds.
WIN_STEP = 0.010  # The step between successive windows in seconds.
NUM_FEATURES = 80  # Number of features to extract.


def input_fn_generator(target):
    """
    Generate the `input_fn` for the TensorFlow estimator.

    Args:
        target (str): The type of input, this affects the used CSV file, batching method and epochs.
            Supported targets are:
            * 'train_bucket': Creates 1 epoch of training data, using bucketing.
                Examples are shuffled.
            * 'train_batch': Creates 1 epoch of training data, using batches.
                Examples are in the order of the `train.csv` file.
            * 'dev': Creates 1 epoch of evaluation data from the `dev.csv` file.
                Uses buckets. Examples are shuffled.
            * 'test': Creates 1 epoch of evaluation data from the `test.csv` file.
                Uses buckets. Examples are shuffled.

    Returns:
        function: Input function pointer.
    """
    if target == 'train_bucket':
        csv_path = FLAGS.train_csv
        use_buckets = True
        epochs = 1
    elif target == 'train_batch':
        csv_path = FLAGS.train_csv
        use_buckets = False
        epochs = 1
    elif target == 'dev':
        csv_path = FLAGS.dev_csv
        use_buckets = True
        epochs = 1
    elif target == 'test':
        csv_path = FLAGS.test_csv
        use_buckets = True
        epochs = 1
    else:
        raise ValueError('Invalid target: "{}"'.format(target))

    def input_fn():
        # L8ER: Try out the following two (not working as of TF v1.12):
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/latency_stats
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/StatsAggregator

        def element_length_fn(_spectrogram, _spectrogram_length, _label_encoded, _label_plaintext):
            del _spectrogram
            del _label_encoded
            del _label_plaintext
            return _spectrogram_length

        assert os.path.exists(csv_path) and os.path.isfile(csv_path)

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_generator(
                __input_generator,
                (tf.float32, tf.int32, tf.int32, tf.string),
                (tf.TensorShape([None, 80]), tf.TensorShape([]),
                 tf.TensorShape([None]), tf.TensorShape([])),
                args=[csv_path, use_buckets])

            if use_buckets:
                # Set shuffle buffer to an arbitrary size to ensure good enough shuffling.
                # At the moment, most shuffling is done by the `__input_generator` function.
                # Also see: https://stackoverflow.com/a/47025850/2785397
                dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

                dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                    element_length_func=element_length_fn,
                    bucket_boundaries=BOUNDARIES,
                    bucket_batch_sizes=[FLAGS.batch_size] * (len(BOUNDARIES) + 1),
                    pad_to_bucket_boundary=False,
                    no_padding=False))

            else:
                dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
                                               padded_shapes=([None, 80], [], [None], []),
                                               drop_remainder=True)

            # dataset.cache()
            dataset = dataset.prefetch(64)

            # Number of epochs.
            dataset = dataset.repeat(epochs)

            iterator = dataset.make_one_shot_iterator()
            spectrogram, spectrogram_length, label_encoded, label_plaintext = iterator.get_next()

            features = {
                'spectrogram': spectrogram,
                'spectrogram_length': spectrogram_length,
                'label_plaintext': label_plaintext
            }

            return features, label_encoded

    return input_fn


def __input_generator(*args):
    assert len(args) == 2, '__input_generator() arguments are a path and shuffle boolean.'
    assert isinstance(args[0], bytes)
    assert isinstance(args[1], np.bool_)
    csv_path = str(args[0], 'utf-8')
    shuffle = bool(args[1])

    with open(csv_path, 'r', encoding='utf-8') as file_handle:
        reader = csv.DictReader(file_handle, delimiter=CSV_DELIMITER, fieldnames=CSV_FIELDNAMES)
        lines = list(reader)[1: -1]  # Remove CSV header and final blank line.

        # Shuffle the CSV lines.
        if shuffle:
            random.shuffle(lines)

        # Read the CSV lines and extract spectrogram and label for each line.
        for line in lines:
            path = line[CSV_HEADER_PATH]
            label = line[CSV_HEADER_LABEL]

            path = os.path.join(CORPUS_DIR, path)

            # Convert the WAV file into
            spectrogram, spectrogram_length = load_sample(path)

            # Convert character sequence label to integer sequence.
            label_encoded = [ctoi(c) for c in label]

            yield spectrogram, spectrogram_length, label_encoded, label


def load_sample(file_path, feature_type=None, feature_normalization=None):
    """
    Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

        feature_type (str): Optional.
            If `None` is provided, use `FLAGS.feature_type`.
            Type of features to generate. Options are 'mel' and 'mfcc'.

        feature_normalization (str): Optional.
            If `None` is provided, use `FLAGS.feature_normalization`.

            Whether to normalize the generated features with the stated method or not.
            Please consult `sample_normalization` for a complete list of normalization methods.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            'none': No normalization is being applied.

    Returns:
        Tuple[np.ndarray. np.ndarray]:
            2D array with [time, num_features] shape, containing `NP_FLOAT`.

            Array containing a single int32.
    """
    __supported_feature_types = ['mel', 'mfcc']
    __supported_feature_normalizations = ['none', 'local', 'local_scalar']

    feature_type = feature_type if feature_type is not None else FLAGS.feature_type
    feature_normalization = feature_normalization if feature_normalization is not None \
        else FLAGS.feature_normalization

    if feature_type not in __supported_feature_types:
        raise ValueError('Requested feature type of {} isn\'t supported.'
                         .format(feature_type))

    if feature_normalization not in __supported_feature_normalizations:
        raise ValueError('Requested feature normalization method {} is invalid.'
                         .format(feature_normalization))

    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate and data.
    (sampling_rate, audio_data) = wavfile.read(file_path)

    if len(audio_data) < 401:
        raise RuntimeError('Sample length {:,d} to short: {}'.format(len(audio_data), file_path))

    if not sampling_rate == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate is {:,d}, expected {:,d}.'
                           .format(sampling_rate, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = WIN_LENGTH  # Window length in ms.
    win_step = WIN_STEP  # Number of milliseconds between successive frames.
    f_max = sampling_rate / 2.  # Maximum frequency (Nyquist rate).
    f_min = 64.  # Minimum frequency.
    n_fft = 1024  # Number of samples in a frame.

    if feature_type == 'mfcc':
        sample = __mfcc(
            audio_data, sampling_rate, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max
        )
    elif feature_type == 'mel':
        sample = __mel(
            audio_data, sampling_rate, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max
        )
    else:
        raise ValueError('Unsupported feature type')

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(NP_FLOAT)

    # Drop every 2nd time frame, if requested.
    if FLAGS.features_drop_every_second_frame:
        # [time, NUM_FEATURES] => [time // 2, NUM_FEATURES]
        sample = sample[:: 2, :]

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Apply feature normalization.
    sample = __feature_normalization(sample, feature_normalization)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def __mfcc(audio_data, sampling_rate, win_len, win_step, num_features, n_fft, f_min, f_max):
    """
    Convert a wav signal into Mel Frequency Cepstral Coefficients (MFCC).

    Args:
        audio_data (np.ndarray): Wav signal.
        sampling_rate (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.

    Returns:
        np.ndarray: MFCC feature vectors. Shape: [time, num_features]
    """
    if num_features % 2 != 0:
        raise ValueError('num_features is not a multiple of 2.')

    # Compute MFCC features.
    mfcc = psf.mfcc(signal=audio_data, samplerate=sampling_rate, winlen=win_len, winstep=win_step,
                    numcep=num_features // 2, nfilt=num_features, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

    # And the first-order differences (delta features).
    mfcc_delta = psf.delta(mfcc, 2)

    # Combine MFCC with MFCC_delta
    return np.concatenate([mfcc, mfcc_delta], axis=1)


def __mel(audio_data, sampling_rate, win_len, win_step, num_features, n_fft, f_min, f_max):
    """
    Convert a wav signal into a logarithmically scaled mel filterbank.

    Args:
        audio_data (np.ndarray): Wav signal.
        sampling_rate (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.

    Returns:
        np.ndarray: Mel-filterbank. Shape: [time, num_features]
    """
    mel = psf.logfbank(signal=audio_data, samplerate=sampling_rate, winlen=win_len,
                       winstep=win_step, nfilt=num_features, nfft=n_fft,
                       lowfreq=f_min, highfreq=f_max, preemph=0.97)
    return mel


def __feature_normalization(features, method):
    """
    Normalize the given feature vector `y`, with the stated normalization `method`.

    Args:
        features (np.ndarray):
            The signal array

        method (str):
            Normalization method:

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            'none': No normalization is being applied.

    Returns:
        np.ndarray: The normalized feature vector.
    """
    if method == 'none':
        return features
    if method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    if method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)

    raise ValueError('Invalid normalization method.')


# Create a dataset for testing purposes.
if __name__ == '__main__':
    __NEXT_ELEMENT = input_fn_generator('train_bucket')

    with tf.Session() as session:
        # for example in range(FLAGS.num_examples_train):
        for example in range(5):
            print('Dataset elements:', session.run(__NEXT_ELEMENT))

    print('The End.')
