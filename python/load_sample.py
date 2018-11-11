"""Helper methods to load audio files."""

import os

import numpy as np
import python_speech_features as psf
from scipy.io import wavfile

from python.params import FLAGS, NP_FLOAT


NUM_FEATURES = 80        # Number of features to extract.
WIN_STEP = 0.010         # The step between successive windows in seconds.


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

            'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            'none': No normalization is being applied.

    Returns:
        Tuple[np.ndarray. np.ndarray]:
            2D array with [time, num_features] shape, containing float.

            Array containing a single int32.
    """
    __supported_feature_types = ['mel', 'mfcc']
    __supported_feature_normalizations = ['none', 'global', 'local', 'local_scalar']

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

    # Load the audio files sample rate (`sr`) and data (`y`).
    (sr, y) = wavfile.read(file_path)

    if len(y) < 401:
        raise RuntimeError('Sample length {:,d} to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate is {:,d}, expected {:,d}.'
                           .format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025         # Window length in ms.
    win_step = WIN_STEP     # Number of milliseconds between successive frames.
    f_max = sr / 2.         # Maximum frequency (Nyquist rate).
    f_min = 64.             # Minimum frequency.
    n_fft = 1024            # Number of samples in a frame.

    if feature_type == 'mfcc':
        sample = _mfcc(y, sr, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max)
    elif feature_type == 'mel':
        sample = _mel(y, sr, win_len, win_step, NUM_FEATURES, n_fft, f_min, f_max)
    else:
        raise ValueError('Unsupported feature type')

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(NP_FLOAT)

    # Drop every 2nd time frame, if requested.
    if FLAGS.features_drop_every_second_frame:
        sample = sample[:: 2, :]

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Apply feature normalization.
    sample = _feature_normalization(sample, feature_normalization)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def _mfcc(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """
    Convert a wav signal into Mel Frequency Cepstral Coefficients (MFCC).

    Args:
        y (np.ndarray): Wav signal.
        sr (int):  Sampling rate.
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
    mfcc = psf.mfcc(signal=y, samplerate=sr, winlen=win_len, winstep=win_step,
                    numcep=num_features // 2, nfilt=num_features, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

    # And the first-order differences (delta features).
    mfcc_delta = psf.delta(mfcc, 2)

    # Combine MFCC with MFCC_delta
    return np.concatenate([mfcc, mfcc_delta], axis=1)


def _mel(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """
    Convert a wav signal into a logarithmically scaled mel filterbank.

    Args:
        y (np.ndarray): Wav signal.
        sr (int):  Sampling rate.
        win_len (float): Window length in seconds.
        win_step (float): Window stride in seconds.
        num_features (int): Number of features to generate.
        n_fft (int): Number of Fast Fourier Transforms.
        f_min (float): Minimum frequency to consider.
        f_max (float): Maximum frequency to consider.

    Returns:
        np.ndarray: Mel-filterbank. Shape: [time, num_features]
    """
    mel = psf.logfbank(signal=y, samplerate=sr, winlen=win_len,
                       winstep=win_step, nfilt=num_features, nfft=n_fft,
                       lowfreq=f_min, highfreq=f_max, preemph=0.97)
    return mel


def _feature_normalization(features, method):
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
    elif method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    elif method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)
    else:
        raise ValueError('Invalid normalization method.')
