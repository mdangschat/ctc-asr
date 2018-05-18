"""Helper methods to load audio files."""

from os import path
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from python.params import FLAGS, NP_FLOAT


NUM_MFCC = 29           # Number of MFCC features to extract.
__WIN_STEP = 0.010      # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean = [5.542525, -3.9271812, -5.6456695, 4.4572716, -4.9022646, -5.9256926, -6.3959913,
                 -6.461323, -2.2669787, -0.65380096, -1.8583168, -1.3298775, -1.9791591,
                 -0.0013444357, -0.0014212646, 0.00041854507, -8.154545e-05, 0.0011094797,
                 0.0034053407, 0.0010864179, 0.0017922536, 0.00059169036, 0.00057138246,
                 0.00053512416, 0.00025640224, 0.00076659914]
__global_mean = np.array(__global_mean, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])

__global_std = [6.99935, 14.969187, 12.69888, 14.14844, 13.50252, 13.42672, 13.453245, 12.993643,
                11.983593, 11.735456, 10.524341, 10.057626, 9.19024, 0.61269754, 3.6294684,
                2.9271908, 3.0644813, 3.134199, 3.2718139, 3.2308598, 3.3299055, 3.1096613,
                3.067856, 2.7884452, 2.625627, 2.4543297]
__global_std = np.array(__global_std, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])


def load_sample(file_path, normalize_features='global', normalize_signal=False):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

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

    if not path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # Load the audio files sample rate (`sr`) and data (`y`)
    (sr, y) = wav.read(file_path)

    if normalize_signal:
        y = signal_normalization(y)

    if len(y) < 401:
        raise RuntimeError('Sample length () to short: {}'.format(len(y), file_path))

    if not sr == FLAGS.sampling_rate:
        raise RuntimeError('Sampling rate of {} found, expected {}.'
                           .format(sr, FLAGS.sampling_rate))

    # At 16000 Hz, 512 samples ~= 32ms. At 16000 Hz, 200 samples = 12ms. 16 samples = 1ms @ 16kHz.
    win_len = 0.025      # Window length in ms.
    win_step = __WIN_STEP  # Number of milliseconds between successive frames.
    f_max = sr / 2.      # Maximum frequency (Nyquist rate).
    f_min = 64.          # Minimum frequency.
    n_fft = 512          # Number of samples in a frame.
    n_mfcc = NUM_MFCC    # Number of Mel cepstral coefficients to extract.
    n_filters = 26       # Number of Mel bins to generate.

    # Compute MFCC features from the mel spectrogram.
    mfcc = psf.mfcc(signal=y, samplerate=sr, winlen=win_len, winstep=win_step,
                    numcep=n_mfcc, nfilt=n_filters, nfft=n_fft,
                    lowfreq=f_min, highfreq=f_max,
                    preemph=0.97, ceplifter=22, appendEnergy=True)

    # And the first-order differences (delta features).
    mfcc_delta = psf.delta(mfcc, 2)

    # Combine MFCC with MFCC_delta
    sample = np.concatenate([mfcc, mfcc_delta], axis=1)

    # Data type.
    sample = sample.astype(NP_FLOAT)

    # Get length of the sample.
    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Sample normalization.
    sample = sample_normalization(sample, normalize_features)

    # `sample` = [time, num_features], `sample_len`: scalar
    return sample, sample_len


def wav_length(file_path):
    """Return time units for a given audio file, corresponding to the number of units if
    the actual features from `load_sample()` where computed.

    Args:
        file_path: Audio file path.

    Returns:
        np.ndarray: Number of time steps of feature extracted audio file.
    """
    if type(file_path) is not str:
        file_path = str(file_path, 'utf-8')

    if not path.isfile(file_path):
        raise ValueError('"{}" does not exist.'.format(file_path))

    # By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
    (sr, y) = wav.read(file_path)

    if not sr == FLAGS.sampling_rate:
        raise TypeError('Sampling rate of {} found, expected {}.'.format(sr, FLAGS.sampling_rate))

    # Load audio data from drive.
    (sr, y) = wav.read(file_path)

    return np.array(int(len(y) / sr / __WIN_STEP), dtype=np.int32)


def signal_normalization(y):
    """Normalize signal by dividing it by its Root Mean Square.
    Formula from:
    <https://dsp.stackexchange.com/questions/26396/normalization-of-a-signal-in-matlab>

    Args:
        y (np.ndarray): The signal data.

    Returns:
        np.ndarray: 1D normalized signal.
    """
    # TODO: RuntimeWarning: invalid value encountered in sqrt
    return y / np.sqrt(np.sum(np.fabs(y) ** 2) / y.shape[0])


def sample_normalization(features, method):
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
