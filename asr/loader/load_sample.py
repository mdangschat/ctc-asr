"""Helper methods to load audio files."""

import os

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from asr.params import FLAGS, NP_FLOAT


NUM_FEATURES = 80        # Number of features to extract.
WIN_STEP = 0.010         # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean = [7.9243336, 8.361284, 8.670752, 8.670842, 8.672162, 8.677621, 8.706355, 8.713961,
                 8.718335, 8.726298, 8.744684, 8.773079, 8.799464, 8.814083, 8.890361, 8.891412,
                 8.8830185, 8.859109, 8.826544, 8.854082, 8.773816, 8.754149, 8.777359, 8.762887,
                 8.785242, 8.776609, 8.742135, 8.770945, 8.772398, 8.774782, 8.809978, 8.817979,
                 8.833887, 8.84898, 8.86401, 8.876555, 8.924455, 8.959041, 8.9077835, 8.890153,
                 8.905932, 8.917205, 8.899732, 8.898879, 8.911425, 8.932618, 9.027659, 9.039076,
                 8.987031, 8.996345, 9.002943, 8.985725, 8.975553, 8.993395, 9.016006, 9.013573,
                 9.008756, 9.019056, 9.001324, 8.984209, 9.0072775, 8.981615, 8.958517, 8.955399,
                 8.949215, 8.944728, 8.926987, 8.909838, 8.91103, 8.910141, 8.907375, 8.907782,
                 8.895392, 8.8923855, 8.885564, 8.875598, 8.867411, 8.852847, 8.826547, 8.732476]
__global_mean = np.array(__global_mean, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])

__global_std = [3.7462842, 3.9170437, 4.0562057, 4.168773, 4.2548923, 4.3175025, 4.3350835,
                4.321269, 4.3271365, 4.375517, 4.427217, 4.472251, 4.490794, 4.497506, 4.519693,
                4.487215, 4.450158, 4.4152765, 4.3585114, 4.3232512, 4.2455106, 4.1881123,
                4.1529436, 4.0925727, 4.0621014, 4.0246553, 3.9794097, 3.99715, 3.9876404,
                3.9816303, 3.9892342, 3.9808958, 3.9113214, 3.9612863, 4.0275817, 4.0417366,
                4.069821, 4.0894203, 4.0489035, 4.029268, 4.032137, 4.0233526, 4.016173, 4.027413,
                4.0482183, 4.076553, 4.1501617, 4.1683154, 4.14113, 4.1289554, 4.1203475, 4.0983853,
                4.0887847, 4.101616, 4.1044135, 4.0944734, 4.0827184, 4.0790124, 4.0542254,
                4.017102, 3.9846838, 3.9519827, 3.9324913, 3.9093099, 3.8885038, 3.8643432,
                3.834046, 3.8254645, 3.8187695, 3.8077822, 3.7993772, 3.7847962, 3.7740924,
                3.7706566, 3.7566462, 3.7314725, 3.7109857, 3.709697, 3.6774132, 3.5467823]
__global_std = np.array(__global_std, dtype=NP_FLOAT).reshape([1, NUM_FEATURES])


def load_sample(file_path, feature_type='mel', normalize_features='local', normalize_signal=False):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

        feature_type (str): Type of features to generate. Options are `'mel'` and `'mfcc'`.

        normalize_features (str or bool):
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

            False: No normalization is being applied.

        normalize_signal (bool):
            Whether to apply (`True`) RMS normalization on the wav signal or not.

    Returns:
        np.ndarray:
            2D array with [time, num_features] shape, containing float.
        np.ndarray:
            Array containing a single int32.
    """
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not os.path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate (`sr`) and data (`y`)
    (sr, y) = wav.read(file_path)

    if normalize_signal:
        y = _signal_normalization(y)

    if len(y) < 401:
        raise RuntimeError('Sample length () to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate of {} found, expected {}.'
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
        raise ValueError('Unsupported feature type "{}".'.format(feature_type))

    # Make sure that data type matches TensorFlow type.
    sample = sample.astype(NP_FLOAT)

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Sample normalization.
    sample = _sample_normalization(sample, normalize_features)

    # sample = [time, NUM_FEATURES], sample_len: scalar
    return sample, sample_len


def _mfcc(y, sr, win_len, win_step, num_features, n_fft, f_min, f_max):
    """Convert a wav signal into Mel Frequency Cepstral Coefficients.

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
        raise ValueError('num_features not a multiple of 2.')

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
    """Convert a wav signal into a logarithmically scaled mel filterbank.

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


def _signal_normalization(y):
    """Normalize signal by dividing it by its Root Mean Square.

    Formula from:
    <https://dsp.stackexchange.com/questions/26396/normalization-of-a-signal-in-matlab>

    TODO: RuntimeWarning: invalid value encountered in sqrt.

    Args:
        y (np.ndarray): The signal data.

    Returns:
        np.ndarray: 1D normalized signal.
    """
    return y / np.sqrt(np.sum(np.fabs(y) ** 2) / y.shape[0])


def _sample_normalization(features, method):
    """Normalize the given feature vector `y`, with the stated normalization `method`.

    Args:
        features (np.ndarray): The signal array
        method (str or bool): Normalization method.

            'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

            'local': Use local (in sample) mean and standard deviation values, and apply the
                normalization element wise, like in `global`.

            'local_scalar': Uses only the mean and standard deviation of the current sample.
                The normalization is being applied by ([sample] - mean_scalar) / std_scalar

            False: No normalization is being applied.

    Returns:
        np.ndarray: The normalized feature vector.
    """
    if not method:
        return features
    elif method == 'global':
        # Option 'global' is applied element wise.
        return (features - __global_mean) / __global_std
    elif method == 'local':
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    elif method == 'local_scalar':
        # Option 'local' uses scalar values.
        return (features - np.mean(features)) / np.std(features)
    else:
        raise ValueError('Invalid normalization method "{}".'.format(method))
