"""Helper methods to load audio files."""

from os import path
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as psf

from python.params import FLAGS, NP_FLOAT


NUM_MFCC = 13           # Number of MFCC features to extract.
__WIN_STEP = 0.0125     # The step between successive windows in seconds.

# Mean and standard deviation values for normalization, according to `sd_estimator.py`.
__global_mean = [5.53072, -4.129051, -5.6504126, 4.540432, -4.8545675, -5.8160844, -6.2034,
                 -6.325581, -2.2667503, -0.65439355, -1.8615158, -1.3312981, -1.9747268,
                 -0.0013710762, -0.0014467032, 0.00040519686, -0.000103923936, 0.0011232576,
                 0.0034771718, 0.0011236193, 0.0018236473, 0.000598786, 0.0005832029,
                 0.0005433287, 0.000266594, 0.00078069617]
__global_mean = np.array(__global_mean, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])

__global_std = [6.96959, 14.574528, 12.609958, 14.011562, 13.47211, 13.417353, 13.422076, 12.961798,
                11.911209, 11.70496, 10.502284, 10.01195, 9.194318, 0.60037535, 3.6021419,
                2.9033747, 3.0421114, 3.1100776, 3.2515237, 3.2137492, 3.3154788, 3.0945556,
                3.061127, 2.7794676, 2.6139154, 2.4488971]
__global_std = np.array(__global_std, dtype=NP_FLOAT).reshape([1, NUM_MFCC * 2])


def load_sample(file_path, normalize='global', normalize_signal=True):
    """Loads the wave file and converts it into feature vectors.

    Args:
        file_path (str or bytes):
            A TensorFlow queue of file names to read from.
            `tf.py_func` converts the provided Tensor into `np.ndarray`s bytes.

        normalize (str or bool):
            Whether to normalize the generated features or not. Supported types are:

                'global': Uses global mean and standard deviation values from `train.txt`.
                The normalization is being applied element wise.
                ([sample] - [mean]^T) / [std]^T
                Where brackets denote matrices or vectors.

                'local': Uses only the mean and standard deviation of the current sample.
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

    # Skip every 2nd time frame.
    sample = sample[:: 2, :]

    sample_len = np.array(sample.shape[0], dtype=np.int32)

    # Sample normalization.
    if normalize is None:
        pass
    elif normalize == 'global':
        # Option 'global' is applied element wise.
        sample = (sample - __global_mean) / __global_std
    elif normalize == 'local':
        # Option 'local' uses scalar values.
        sample = (sample - np.mean(sample)) / np.std(sample)
    else:
        raise ValueError('Invalid normalization method "{}".'.format(normalize))

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

    # The /2 is because `load_sample` skips every 2nd frame.
    return np.array(int(len(y) / sr / __WIN_STEP) // 2, dtype=np.int32)


def signal_normalization(y):
    """Normalize signal by dividing it by its Root Mean Square.
    Formula taken from:
    <https://dsp.stackexchange.com/questions/26396/normalization-of-a-signal-in-matlab>

    Args:
        y (np.ndarray): The signal data.

    Returns:
        np.ndarray: 1D normalized signal.
    """
    return y / np.sqrt(np.sum(np.abs(y) ** 2) / y.shape[0])
